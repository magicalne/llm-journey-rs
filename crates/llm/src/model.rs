use snafu::prelude::*;
use snafu::{ResultExt, Snafu};
use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use std::mem;
use std::path::Path;

#[derive(Debug, Snafu)]
pub enum Error {
    #[snafu(display("Failed to read u8: {}", source))]
    ReadU8 { source: io::Error },
    #[snafu(display("Failed to read i8: {}", source))]
    ReadI8 { source: io::Error },
    #[snafu(display("Failed to read u16: {}", source))]
    ReadU16 { source: io::Error },
    #[snafu(display("Failed to read i16: {}", source))]
    ReadI16 { source: io::Error },
    #[snafu(display("Failed to read u32: {}", source))]
    ReadU32 { source: io::Error },
    #[snafu(display("Failed to read i32: {}", source))]
    ReadI32 { source: io::Error },
    #[snafu(display("Failed to read u64: {}", source))]
    ReadU64 { source: io::Error },
    #[snafu(display("Failed to read i64: {}", source))]
    ReadI64 { source: io::Error },
    #[snafu(display("Failed to read f32: {}", source))]
    ReadF32 { source: io::Error },
    #[snafu(display("Failed to read f64: {}", source))]
    ReadF64 { source: io::Error },
    #[snafu(display("Failed to read bool: {}", source))]
    ReadBool { source: io::Error },
    #[snafu(display("Invalid metadata value type"))]
    InvalidMetadataValueType,
    #[snafu(display("Invalid tensor type"))]
    InvalidTensorType,
    #[snafu(display("Failed to read exact: {}", source))]
    ReadExact { source: io::Error },
    #[snafu(display("Failed to read string: {}", source))]
    ReadString { source: io::Error },
    #[snafu(display("Failed to read GgufString: {}", source))]
    ReadGgufString { source: io::Error },
    #[snafu(display("Failed to read GgufMetadataKv: {}", source))]
    ReadGgufMetadataKv { source: io::Error },
    #[snafu(display("Failed to read GgufMetadataValue: {}", source))]
    ReadGgufMetadataValue { source: io::Error },
    #[snafu(display("Failed to read GgufHeader: {}", source))]
    ReadGgufHeader { source: io::Error },
    #[snafu(display("Failed to read GgufTensorInfo: {}", source))]
    ReadGgufTensorInfo { source: io::Error },
    #[snafu(display("Failed to read GgufFile: {}", source))]
    ReadGgufFile { source: io::Error },
}

type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug)]
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_K,
    IQ2_XXS,
    IQ2_XS,
    IQ3_XXS,
    IQ1_S,
    IQ4_NL,
    IQ3_S,
    IQ2_S,
    IQ4_XS,
    I8,
    I16,
    I32,
    I64,
    F64,
    IQ1_M,
}

#[derive(Debug, Clone)]
pub enum GgufMetadataValueType {
    Uint8,
    Int8,
    Uint16,
    Int16,
    Uint32,
    Int32,
    Float32,
    Bool,
    String,
    Array,
    Uint64,
    Int64,
    Float64,
}

#[derive(Debug)]
pub struct GgufString {
    len: u64,
    string: String,
}

#[derive(Debug)]
pub enum GgufMetadataValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
    Bool(bool),
    String(GgufString),
    Array(Vec<GgufMetadataValue>),
}

#[derive(Debug)]
pub struct GgufMetadataKv {
    key: GgufString,
    value_type: GgufMetadataValueType,
    value: GgufMetadataValue,
}

#[derive(Debug)]
pub struct GgufHeader {
    magic: u32,
    version: u32,
    tensor_count: u64,
    metadata_kv_count: u64,
    metadata_kv: Vec<GgufMetadataKv>,
}

#[derive(Debug)]
pub struct GgufTensorInfo {
    name: GgufString,
    n_dimensions: u32,
    dimensions: Vec<u64>,
    tensor_type: GgmlType,
    offset: u64,
}

#[derive(Debug)]
pub struct GgufFile {
    header: GgufHeader,
    tensor_infos: Vec<GgufTensorInfo>,
    tensor_data: Vec<u8>,
}

impl GgufFile {
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path).context(ReadGgufFileSnafu)?;
        let mut reader = BufReader::new(file);

        let header = GgufHeader::read(&mut reader)?;
        let tensor_infos = (0..header.tensor_count)
            .map(|_| GgufTensorInfo::read(&mut reader))
            .collect::<Result<Vec<_>>>()?;

        let mut tensor_data = Vec::new();

        reader
            .seek(SeekFrom::Start(
                header.tensor_count * mem::size_of::<GgufTensorInfo>() as u64,
            ))
            .context(ReadGgufFileSnafu)?;

        reader
            .read_to_end(&mut tensor_data)
            .context(ReadGgufFileSnafu)?;

        Ok(GgufFile {
            header,
            tensor_infos,
            tensor_data,
        })
    }
}

impl GgufHeader {
    fn read<R: Read>(reader: &mut R) -> Result<Self> {
        let magic = read_u32(reader)?;

        let version = read_u32(reader)?;
        let tensor_count = read_u64(reader)?;
        let metadata_kv_count = read_u64(reader)?;

        println!("kv cnt: {}", metadata_kv_count);
        let metadata_kv = (0..metadata_kv_count)
            .map(|_| GgufMetadataKv::read(reader))
            .collect::<Result<Vec<_>>>()?;

        Ok(GgufHeader {
            magic,
            version,
            tensor_count,
            metadata_kv_count,
            metadata_kv,
        })
    }
}

impl GgufMetadataKv {
    fn read<R: Read>(reader: &mut R) -> Result<Self> {
        let key = GgufString::read(reader)?;
        let value_type = match read_u32(reader)? {
            0 => GgufMetadataValueType::Uint8,
            1 => GgufMetadataValueType::Int8,
            2 => GgufMetadataValueType::Uint16,
            3 => GgufMetadataValueType::Int16,
            4 => GgufMetadataValueType::Uint32,
            5 => GgufMetadataValueType::Int32,
            6 => GgufMetadataValueType::Float32,
            7 => GgufMetadataValueType::Bool,
            8 => GgufMetadataValueType::String,
            9 => GgufMetadataValueType::Array,
            10 => GgufMetadataValueType::Uint64,
            11 => GgufMetadataValueType::Int64,
            12 => GgufMetadataValueType::Float64,
            _ => return Err(Error::InvalidMetadataValueType),
        };
        println!("test111: {:?}", value_type);
        let value = GgufMetadataValue::read(reader, value_type.clone())?;
        println!("test222: {:?}", value_type);

        Ok(GgufMetadataKv {
            key,
            value_type,
            value,
        })
    }
}

impl GgufMetadataValue {
    fn read<R: Read>(reader: &mut R, value_type: GgufMetadataValueType) -> Result<Self> {
        match value_type {
            GgufMetadataValueType::Uint8 => Ok(GgufMetadataValue::Uint8(read_u8(reader)?)),
            GgufMetadataValueType::Int8 => Ok(GgufMetadataValue::Int8(read_i8(reader)?)),
            GgufMetadataValueType::Uint16 => Ok(GgufMetadataValue::Uint16(read_u16(reader)?)),
            GgufMetadataValueType::Int16 => Ok(GgufMetadataValue::Int16(read_i16(reader)?)),
            GgufMetadataValueType::Uint32 => Ok(GgufMetadataValue::Uint32(read_u32(reader)?)),
            GgufMetadataValueType::Int32 => Ok(GgufMetadataValue::Int32(read_i32(reader)?)),
            GgufMetadataValueType::Float32 => Ok(GgufMetadataValue::Float32(read_f32(reader)?)),
            GgufMetadataValueType::Bool => Ok(GgufMetadataValue::Bool(read_bool(reader)?)),
            GgufMetadataValueType::String => {
                Ok(GgufMetadataValue::String(GgufString::read(reader)?))
            }
            GgufMetadataValueType::Array => {
                let len = read_u64(reader)?;
                println!("len: {}", len);
                let mut array = Vec::with_capacity(len as usize);
                for _ in 0..len {
                    let value_type = match read_u32(reader)? {
                        0 => GgufMetadataValueType::Uint8,
                        1 => GgufMetadataValueType::Int8,
                        2 => GgufMetadataValueType::Uint16,
                        3 => GgufMetadataValueType::Int16,
                        4 => GgufMetadataValueType::Uint32,
                        5 => GgufMetadataValueType::Int32,
                        6 => GgufMetadataValueType::Float32,
                        7 => GgufMetadataValueType::Bool,
                        8 => GgufMetadataValueType::String,
                        9 => GgufMetadataValueType::Array,
                        10 => GgufMetadataValueType::Uint64,
                        11 => GgufMetadataValueType::Int64,
                        12 => GgufMetadataValueType::Float64,
                        _ => return Err(Error::InvalidMetadataValueType),
                    };
                    array.push(GgufMetadataValue::read(reader, value_type)?);
                }

                Ok(GgufMetadataValue::Array(array))
            }
            GgufMetadataValueType::Uint64 => Ok(GgufMetadataValue::Uint64(read_u64(reader)?)),
            GgufMetadataValueType::Int64 => Ok(GgufMetadataValue::Int64(read_i64(reader)?)),
            GgufMetadataValueType::Float64 => Ok(GgufMetadataValue::Float64(read_f64(reader)?)),
        }
    }
}

impl GgufString {
    fn read<R: Read>(reader: &mut R) -> Result<Self> {
        let len = read_u64(reader)?;
        let mut string = vec![0; len as usize];
        reader.read_exact(&mut string).context(ReadExactSnafu)?;
        let string = String::from_utf8(string)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
            .context(ReadStringSnafu)?;
        Ok(GgufString { len, string })
    }
}

impl GgufTensorInfo {
    fn read<R: Read>(reader: &mut R) -> Result<Self> {
        let name = GgufString::read(reader)?;
        let n_dimensions = read_u32(reader)?;
        let dimensions = (0..n_dimensions)
            .map(|_| read_u64(reader))
            .collect::<Result<Vec<_>>>()?;
        let tensor_type = match read_u32(reader)? {
            0 => GgmlType::F32,
            1 => GgmlType::F16,
            2 => GgmlType::Q4_0,
            3 => GgmlType::Q4_1,
            6 => GgmlType::Q5_0,
            7 => GgmlType::Q5_1,
            8 => GgmlType::Q8_0,
            9 => GgmlType::Q8_1,
            10 => GgmlType::Q2_K,
            11 => GgmlType::Q3_K,
            12 => GgmlType::Q4_K,
            13 => GgmlType::Q5_K,
            14 => GgmlType::Q6_K,
            15 => GgmlType::Q8_K,
            16 => GgmlType::IQ2_XXS,
            17 => GgmlType::IQ2_XS,
            18 => GgmlType::IQ3_XXS,
            19 => GgmlType::IQ1_S,
            20 => GgmlType::IQ4_NL,
            21 => GgmlType::IQ3_S,
            22 => GgmlType::IQ2_S,
            23 => GgmlType::IQ4_XS,
            24 => GgmlType::I8,
            25 => GgmlType::I16,
            26 => GgmlType::I32,
            27 => GgmlType::I64,
            28 => GgmlType::F64,
            29 => GgmlType::IQ1_M,
            _ => return Err(Error::InvalidTensorType),
        };
        let offset = read_u64(reader)?;

        Ok(GgufTensorInfo {
            name,
            n_dimensions,
            dimensions,
            tensor_type,
            offset,
        })
    }
}

fn read_u8<R: Read>(reader: &mut R) -> Result<u8> {
    let mut buf = [0; 1];
    reader.read_exact(&mut buf).context(ReadU8Snafu)?;
    Ok(buf[0])
}

fn read_i8<R: Read>(reader: &mut R) -> Result<i8> {
    let mut buf = [0; 1];
    reader.read_exact(&mut buf).context(ReadI8Snafu)?;
    Ok(buf[0] as i8)
}

fn read_u16<R: Read>(reader: &mut R) -> Result<u16> {
    let mut buf = [0; 2];
    reader.read_exact(&mut buf).context(ReadU16Snafu)?;
    Ok(u16::from_le_bytes(buf))
}

fn read_i16<R: Read>(reader: &mut R) -> Result<i16> {
    let mut buf = [0; 2];
    reader.read_exact(&mut buf).context(ReadI16Snafu)?;
    Ok(i16::from_le_bytes(buf))
}

fn read_u32<R: Read>(reader: &mut R) -> Result<u32> {
    let mut buf = [0; 4];
    reader.read_exact(&mut buf).context(ReadU32Snafu)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_i32<R: Read>(reader: &mut R) -> Result<i32> {
    let mut buf = [0; 4];
    reader.read_exact(&mut buf).context(ReadI32Snafu)?;
    Ok(i32::from_le_bytes(buf))
}

fn read_u64<R: Read>(reader: &mut R) -> Result<u64> {
    let mut buf = [0; 8];
    reader.read_exact(&mut buf).context(ReadU64Snafu)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_i64<R: Read>(reader: &mut R) -> Result<i64> {
    let mut buf = [0; 8];
    reader.read_exact(&mut buf).context(ReadI64Snafu)?;
    Ok(i64::from_le_bytes(buf))
}

fn read_f32<R: Read>(reader: &mut R) -> Result<f32> {
    let mut buf = [0; 4];
    reader.read_exact(&mut buf).context(ReadF32Snafu)?;
    Ok(f32::from_le_bytes(buf))
}

fn read_f64<R: Read>(reader: &mut R) -> Result<f64> {
    let mut buf = [0; 8];
    reader.read_exact(&mut buf).context(ReadF64Snafu)?;
    Ok(f64::from_le_bytes(buf))
}

fn read_bool<R: Read>(reader: &mut R) -> Result<bool> {
    let mut buf = [0; 1];
    reader.read_exact(&mut buf).context(ReadBoolSnafu)?;
    Ok(buf[0] != 0)
}
