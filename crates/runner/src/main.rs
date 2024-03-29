use arrow::array::{GenericListArray, GenericListBuilder, Int64Builder};
use arrow_schema::ffi::FFI_ArrowSchema;
use arrow_schema::Field;
use datafusion::arrow::array::{ArrayRef, StringArray};
use datafusion::arrow::datatypes::DataType;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::context::SessionContext;
use datafusion::logical_expr::{ScalarUDF, ScalarUDFImpl, Signature};
use libloading::Library;
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::ffi::CStr;
use std::sync::{Arc, RwLock};

#[derive(Debug)]
pub struct FFIPlugin {
    /// Shared library.
    pub lib: Arc<str>,
    /// Identifier in the shared lib.
    pub symbol: Arc<str>,
    /// Pickle serialized keyword arguments.
    pub kwargs: Arc<[u8]>,
    pub signature: Signature,
}
type PluginAndVersion = (Library, u16, u16);
static LOADED: Lazy<RwLock<HashMap<String, PluginAndVersion>>> = Lazy::new(Default::default);

#[repr(C)]
#[derive(Debug)]
pub struct SeriesExport {
    field: *mut arrow_schema::ffi::FFI_ArrowSchema,
    // A double ptr, so we can easily release the buffer
    // without dropping the arrays.
    arrays: *mut *mut arrow::ffi::FFI_ArrowArray,
    len: usize,
    release: Option<unsafe extern "C" fn(arg1: *mut SeriesExport)>,
    private_data: *mut std::os::raw::c_void,
}

// A utility that helps releasing/owning memory.
#[allow(dead_code)]
struct PrivateData {
    schema: Box<arrow_schema::ffi::FFI_ArrowSchema>,
    arrays: Box<[*mut arrow::ffi::FFI_ArrowArray]>,
}

/// # Safety
/// `ArrowArray` and `ArrowSchema` must be valid
unsafe fn import_array(
    array: arrow::ffi::FFI_ArrowArray,
    schema: &arrow_schema::ffi::FFI_ArrowSchema,
) -> Result<ArrayRef> {
    let data = arrow::ffi::from_ffi(array, schema).unwrap();
    let arr = arrow::array::make_array(data);
    Ok(arr)
}

/// # Safety
/// `SeriesExport` must be valid
pub unsafe fn import_series(e: SeriesExport) -> Result<ArrayRef> {
    let schema = FFI_ArrowSchema::from_raw(e.field);

    let pointers = std::slice::from_raw_parts_mut(e.arrays, e.len);
    let chunks = pointers
        .iter()
        .map(|ptr| {
            let arr = std::ptr::read(*ptr);
            import_array(arr, &schema)
        })
        .collect::<Result<Vec<_>>>()?;

    // we only support a single chunk for now.
    let chunk0 = chunks[0].clone();
    Ok(chunk0)
}

// callback used to drop [SeriesExport] when it is exported.
unsafe extern "C" fn c_release_series_export(e: *mut SeriesExport) {
    if e.is_null() {
        return;
    }
    let e = &mut *e;
    e.release = None;
}

impl SeriesExport {
    pub fn empty() -> Self {
        Self {
            field: std::ptr::null_mut(),
            arrays: std::ptr::null_mut(),
            len: 0,
            release: None,
            private_data: std::ptr::null_mut(),
        }
    }

    pub fn is_null(&self) -> bool {
        self.private_data.is_null()
    }

    pub fn from_array(array: ArrayRef) -> Self {
        let data = array.to_data();
        let (ffi_arr, ffi_schema) = arrow::ffi::to_ffi(&data).unwrap();
        let ffi_schema = Box::new(ffi_schema);
        let field = ffi_schema.as_ref() as *const FFI_ArrowSchema as *mut _;
        let mut arrays = Box::new([Box::into_raw(Box::new(ffi_arr))]);
        let len = arrays.len();
        let ptr = arrays.as_mut_ptr();

        Self {
            field,
            arrays: ptr,
            len,
            release: Some(c_release_series_export),
            private_data: Box::into_raw(Box::new(PrivateData {
                arrays,
                schema: ffi_schema,
            })) as *mut std::os::raw::c_void,
        }
    }
}

impl Drop for SeriesExport {
    fn drop(&mut self) {
        if let Some(release) = self.release {
            unsafe { release(self) }
        }
    }
}

/// Passed to an expression.
/// This contains information for the implementer of the expression on what it is allowed to do.
#[derive(Copy, Clone, Debug, Default)]
#[repr(C)]
pub struct CallerContext {
    // bit
    // 1: PARALLEL
    bitflags: u64,
}

impl CallerContext {
    const fn kth_bit_set(&self, k: u64) -> bool {
        (self.bitflags & (1 << k)) > 0
    }

    fn set_kth_bit(&mut self, k: u64) {
        self.bitflags |= 1 << k
    }

    /// Parallelism is done by polars' main engine, the plugin should not run run its own parallelism.
    /// If this is `false`, the plugin could use parallelism without (much) contention with polars
    /// parallelism strategies.
    pub fn parallel(&self) -> bool {
        self.kth_bit_set(0)
    }

    pub fn _set_parallel(&mut self) {
        self.set_kth_bit(0)
    }
}

fn get_lib(lib: &str) -> Result<&'static PluginAndVersion> {
    let lib_map = LOADED.read().unwrap();
    if let Some(library) = lib_map.get(lib) {
        // lifetime is static as we never remove libraries.
        Ok(unsafe { std::mem::transmute::<&PluginAndVersion, &'static PluginAndVersion>(library) })
    } else {
        drop(lib_map);
        let library = unsafe { Library::new(lib).unwrap() };
        let version_function: libloading::Symbol<unsafe extern "C" fn() -> u32> = unsafe {
            library
                .get("_polars_plugin_get_version".as_bytes())
                .unwrap()
        };

        let version = unsafe { version_function() };
        let major = (version >> 16) as u16;
        let minor = version as u16;

        let mut lib_map = LOADED.write().unwrap();
        lib_map.insert(lib.to_string(), (library, major, minor));
        drop(lib_map);

        get_lib(lib)
    }
}
pub unsafe fn plugin_return_type(
    fields: &[Field],
    lib: &str,
    symbol: &str,
    kwargs: &[u8],
) -> Result<Field> {
    let plugin = get_lib(lib)?;
    let lib = &plugin.0;
    let major = plugin.1;
    let minor = plugin.2;

    // we deallocate the fields buffer
    let ffi_fields = fields
        .iter()
        .map(|field| field.try_into().unwrap())
        .collect::<Vec<FFI_ArrowSchema>>()
        .into_boxed_slice();
    let n_args = ffi_fields.len();
    let slice_ptr = ffi_fields.as_ptr();

    let mut return_value = FFI_ArrowSchema::empty();
    let return_value_ptr = &mut return_value as *mut FFI_ArrowSchema;

    if major == 0 {
        match minor {
            0 => {
                // *const ArrowSchema: pointer to heap Box<ArrowSchema>
                // usize: length of the boxed slice
                // *mut ArrowSchema: pointer where the return value can be written
                let symbol: libloading::Symbol<
                    unsafe extern "C" fn(*const FFI_ArrowSchema, usize, *mut FFI_ArrowSchema),
                > = lib
                    .get((format!("_polars_plugin_field_{}", symbol)).as_bytes())
                    .unwrap();
                symbol(slice_ptr, n_args, return_value_ptr);
            }
            1 => {
                // *const FFI_ArrowSchema: pointer to heap Box<FFI_ArrowSchema>
                // usize: length of the boxed slice
                // *mut FFI_ArrowSchema: pointer where the return value can be written
                // *const u8: pointer to &[u8] (kwargs)
                // usize: length of the u8 slice
                let symbol: libloading::Symbol<
                    unsafe extern "C" fn(
                        *const FFI_ArrowSchema,
                        usize,
                        *mut FFI_ArrowSchema,
                        *const u8,
                        usize,
                    ),
                > = lib
                    .get((format!("_polars_plugin_field_{}", symbol)).as_bytes())
                    .unwrap();

                let kwargs_ptr = kwargs.as_ptr();
                let kwargs_len = kwargs.len();

                symbol(slice_ptr, n_args, return_value_ptr, kwargs_ptr, kwargs_len);
            }
            _ => {
                todo!()
            }
        }

        let ret_val_ptr = &return_value as *const FFI_ArrowSchema;
        if !ret_val_ptr.is_null() {
            let out = Field::try_from(&return_value).unwrap();

            Ok(out)
        } else {
            let msg = retrieve_error_msg(lib);
            let msg = msg.to_string_lossy();
            panic!("{}", msg.as_ref());
        }
    } else {
        todo!()
    }
}

unsafe fn retrieve_error_msg(lib: &Library) -> &CStr {
    let symbol: libloading::Symbol<unsafe extern "C" fn() -> *mut std::os::raw::c_char> =
        lib.get(b"_polars_plugin_get_last_error_message\0").unwrap();
    let msg_ptr = symbol();
    CStr::from_ptr(msg_ptr)
}

impl ScalarUDFImpl for FFIPlugin {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn name(&self) -> &str {
        &self.symbol
    }

    fn signature(&self) -> &datafusion::logical_expr::Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> datafusion::error::Result<DataType> {
        let flds = arg_types
            .iter()
            .enumerate()
            .map(|(i, dtype)| Field::new(format!("col_{i}"), dtype.clone(), true))
            .collect::<Vec<_>>();
        let field = unsafe { plugin_return_type(&flds, &self.lib, &self.symbol, &self.kwargs)? };

        Ok(field.data_type().clone())
    }

    fn invoke(
        &self,
        args: &[datafusion::physical_plan::ColumnarValue],
    ) -> datafusion::error::Result<datafusion::physical_plan::ColumnarValue> {
        unsafe {
            let plugin = get_lib(&self.lib)?;
            let lib = &plugin.0;
            let _major = plugin.1;
            let symbol: libloading::Symbol<
                unsafe extern "C" fn(
                    *const SeriesExport,
                    usize,
                    *const u8,
                    usize,
                    *mut SeriesExport,
                    *const CallerContext,
                ),
            > = lib
                .get(format!("_polars_plugin_{}", self.symbol).as_bytes())
                .unwrap();
            let input = args
                .iter()
                .map(|arg| {
                    let input = match arg {
                        datafusion::physical_plan::ColumnarValue::Array(array) => array,
                        _ => todo!(),
                    };
                    SeriesExport::from_array(input.clone())
                })
                .collect::<Vec<_>>();
            let input_len = args.len();
            let slice_ptr = input.as_ptr();

            let kwargs_ptr = self.kwargs.as_ptr();
            let kwargs_len = self.kwargs.len();

            let mut return_value = SeriesExport::empty();
            let return_value_ptr = &mut return_value as *mut SeriesExport;
            let context = CallerContext::default();
            let context_ptr = &context as *const CallerContext;
            symbol(
                slice_ptr,
                input_len,
                kwargs_ptr,
                kwargs_len,
                return_value_ptr,
                context_ptr,
            );

            if !return_value.is_null() {
                let value = import_series(return_value)?;
                Ok(datafusion::physical_plan::ColumnarValue::Array(value))
            } else {
                let msg = retrieve_error_msg(lib);
                let msg = msg.to_string_lossy();
                Err(DataFusionError::Execution(msg.to_string()))
            }
        }
    }
}
fn create_context() -> datafusion::error::Result<SessionContext> {
    let strings: ArrayRef = Arc::new(StringArray::from(vec![
        "The cat sits outside",
        "A man is playing guitar",
        "I love pasta",
    ]));

    // "dist_a": [[12, 32, 1], [], [1, -2]],
    // "dist_b": [[-12, 1], [43], [876, -45, 9]],

    let inner_builder = Int64Builder::new();
    let mut dist_a = GenericListBuilder::new(inner_builder);
    dist_a.append_value(vec![Some(12), Some(32), Some(1)]);
    dist_a.append_value(vec![]);
    dist_a.append_value(vec![Some(1), Some(-2)]);
    let dist_a: GenericListArray<i32> = dist_a.finish();
    let dist_a = Arc::new(dist_a);
    let inner_builder = Int64Builder::new();
    let mut dist_b = GenericListBuilder::new(inner_builder);
    dist_b.append_value(vec![Some(-12), Some(1)]);
    dist_b.append_value(vec![Some(43)]);
    dist_b.append_value(vec![Some(876), Some(-45), Some(9)]);
    let dist_b: GenericListArray<i32> = dist_b.finish();
    let dist_b = Arc::new(dist_b);
    let batch = RecordBatch::try_from_iter(vec![
        ("strings", strings),
        ("dist_a", dist_a),
        ("dist_b", dist_b),
    ])?;

    let ctx = SessionContext::new();

    ctx.register_batch("t", batch)?;
    Ok(ctx)
}

fn main() -> anyhow::Result<()> {
    let runtime = tokio::runtime::Runtime::new()?;
    runtime.block_on(run())
}

#[derive(serde::Serialize, serde::Deserialize, Debug)]
struct PigLatinKwargs {
    capitalize: bool,
}

async fn run() -> anyhow::Result<()> {
    let ctx = create_context()?;
    let target_dir = env!("CARGO_MANIFEST_DIR");
    let lib = format!("{}/../../target/debug/libjaccard_lib.dylib", target_dir);
    let lib = std::path::Path::new(&lib);
    let lib = lib.canonicalize()?;
    let lib = lib.to_str().unwrap();

    let ffi_plugin = FFIPlugin {
        lib: Arc::from(lib),
        symbol: Arc::from("jaccard_similarity"),
        kwargs: Arc::new([]),
        // we know nothing about the signature, only the return type
        signature: Signature::variadic_any(datafusion::logical_expr::Volatility::Volatile),
    };
    let plugin = ScalarUDF::from(ffi_plugin);

    ctx.register_udf(plugin.clone());

    let sql_df = ctx
        .sql("SELECT jaccard_similarity(dist_a, dist_b) FROM t")
        .await?;
    sql_df.show().await?;

    Ok(())
}
