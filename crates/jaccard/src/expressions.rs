use polars::prelude::*;
use polars_plan::dsl::FieldsMapper;
use pyo3_polars::derive::polars_expr;

#[polars_expr(output_type=Float64)]
fn jaccard_similarity(inputs: &[Series]) -> PolarsResult<Series> {
    let a = inputs[0].list()?;
    let b = inputs[1].list()?;
    crate::distances::naive_jaccard_sim(a, b).map(|ca| ca.into_series())
}

#[polars_expr(output_type=Float64)]
fn hamming_distance(inputs: &[Series]) -> PolarsResult<Series> {
    let a = inputs[0].str()?;
    let b = inputs[1].str()?;
    let out: UInt32Chunked =
        arity::binary_elementwise_values(a, b, crate::distances::naive_hamming_dist);
    Ok(out.into_series())
}

fn haversine_output(input_fields: &[Field]) -> PolarsResult<Field> {
    FieldsMapper::new(input_fields).map_to_float_dtype()
}

#[polars_expr(output_type_func=haversine_output)]
fn haversine(inputs: &[Series]) -> PolarsResult<Series> {
    let out = match inputs[0].dtype() {
        DataType::Float32 => {
            let start_lat = inputs[0].f32().unwrap();
            let start_long = inputs[1].f32().unwrap();
            let end_lat = inputs[2].f32().unwrap();
            let end_long = inputs[3].f32().unwrap();
            crate::distances::naive_haversine(start_lat, start_long, end_lat, end_long)?
                .into_series()
        }
        DataType::Float64 => {
            let start_lat = inputs[0].f64().unwrap();
            let start_long = inputs[1].f64().unwrap();
            let end_lat = inputs[2].f64().unwrap();
            let end_long = inputs[3].f64().unwrap();
            crate::distances::naive_haversine(start_lat, start_long, end_lat, end_long)?
                .into_series()
        }
        _ => unimplemented!(),
    };
    Ok(out)
}
