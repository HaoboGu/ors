use ors_sys::*;

/// Trait used to map Rust types (for example `f32`) to ONNX types (for example `Float`)
pub trait TypeToTensorElementDataType {
    /// Return the ONNX type for a Rust type
    fn tensor_element_data_type() -> ONNXTensorElementDataType
    where
        Self: Sized;

    /// If the type is `String`, returns `Some` with utf8 contents, else `None`.
    fn try_utf8_bytes(&self) -> Option<&[u8]>;
}

macro_rules! impl_type_trait {
    ($type_:ty, $variant:ident) => {
        impl TypeToTensorElementDataType for $type_ {
            fn tensor_element_data_type() -> ONNXTensorElementDataType {
                // unsafe { std::mem::transmute(TensorElementDataType::$variant) }
                $variant
            }

            fn try_utf8_bytes(&self) -> Option<&[u8]> {
                None
            }
        }
    };
}

impl_type_trait!(
    f32,
    ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
);
impl_type_trait!(
    u8,
    ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8
);
impl_type_trait!(
    i8,
    ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8
);
impl_type_trait!(
    u16,
    ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16
);
impl_type_trait!(
    i16,
    ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16
);
impl_type_trait!(
    i32,
    ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
);
impl_type_trait!(
    i64,
    ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
);
impl_type_trait!(
    bool,
    ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL
);
// impl_type_trait!(f16, Float16);
impl_type_trait!(
    f64,
    ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE
);
impl_type_trait!(
    u32,
    ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32
);
impl_type_trait!(
    u64,
    ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64
);
// impl_type_trait!(, Complex64);
// impl_type_trait!(, Complex128);
// impl_type_trait!(, Bfloat16);
