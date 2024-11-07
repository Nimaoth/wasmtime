use crate::ForeignData;
use std::println;
use std::ffi::c_void;
use anyhow::{bail, Result, anyhow};
use wasmtime::component::{Component, Func, Instance, Linker, Val};
use wasmtime::{AsContextMut, StoreContextMut, Store, StoreLimits,
    StoreLimitsBuilder};
use wasmtime_wasi::{WasiView, WasiCtx};


use crate::{
    declare_vecs, handle_call_error, handle_result, wasm_byte_vec_t, wasm_config_t, wasm_engine_t,
    wasm_name_t, wasm_trap_t, wasmtime_error_t, WasmtimeStoreData,
};
use std::{mem, mem::MaybeUninit, ptr, slice};

impl TryInto<String> for &wasm_name_t {
    type Error = anyhow::Error;

    fn try_into(self) -> Result<String, Self::Error> {
        let name = std::str::from_utf8(self.as_slice())?;
        Ok(name.to_owned())
    }
}

#[no_mangle]
pub extern "C" fn wasmtime_config_component_model_set(c: &mut wasm_config_t, enable: bool) {
    c.config.wasm_component_model(enable);
}

pub struct Host {
    table: wasmtime::component::ResourceTable,
    ctx: WasiCtx,
    data: WasmtimeStoreData
}

impl WasiView for Host {
    fn table(&mut self) -> &mut wasmtime::component::ResourceTable {
        &mut self.table
    }

    fn ctx(&mut self) -> &mut WasiCtx {
        &mut self.ctx
    }
}

pub type WasmtimeStoreContextMutWasi<'a> = StoreContextMut<'a, Host>;
pub type wasmtime_component_val_string_t = wasm_byte_vec_t;

/// Representation of a `Store` for `wasmtime.h` This notably tries to move more
/// burden of aliasing on the caller rather than internally, allowing for a more
/// raw representation of contexts and such that requires less `unsafe` in the
/// implementation.
///
/// Note that this notably carries `WasmtimeStoreData` as a payload which allows
/// storing foreign data and configuring WASI as well.
#[repr(C)]
pub struct wasmtime_component_store_t {
    pub(crate) store: Store<Host>,
}

wasmtime_c_api_macros::declare_own!(wasmtime_component_store_t);

#[no_mangle]
pub extern "C" fn wasmtime_component_store_new(
    engine: &wasm_engine_t,
    data: *mut c_void,
    finalizer: Option<extern "C" fn(*mut c_void)>,
) -> Box<wasmtime_component_store_t> {
    Box::new(wasmtime_component_store_t {
        store: Store::new(
            &engine.engine,
            Host {
                table: wasmtime::component::ResourceTable::new(),
                ctx: WasiCtx::builder().inherit_stdio().build(),
                data: WasmtimeStoreData {
                    foreign: ForeignData { data, finalizer },
                    #[cfg(feature = "wasi")]
                    wasi: None,
                    hostcall_val_storage: Vec::new(),
                    wasm_val_storage: Vec::new(),
                    store_limits: StoreLimits::default(),
                },
            },
        ),
    })
}

#[no_mangle]
pub extern "C" fn wasmtime_component_store_context(
    store: &mut wasmtime_component_store_t,
) -> WasmtimeStoreContextMutWasi<'_> {
    store.store.as_context_mut()
}

#[no_mangle]
pub extern "C" fn wasmtime_component_store_limiter(
    store: &mut wasmtime_component_store_t,
    memory_size: i64,
    table_elements: i64,
    instances: i64,
    tables: i64,
    memories: i64,
) {
    let mut limiter = StoreLimitsBuilder::new();
    if memory_size >= 0 {
        limiter = limiter.memory_size(memory_size as usize);
    }
    if table_elements >= 0 {
        limiter = limiter.table_elements(table_elements as usize);
    }
    if instances >= 0 {
        limiter = limiter.instances(instances as usize);
    }
    if tables >= 0 {
        limiter = limiter.tables(tables as usize);
    }
    if memories >= 0 {
        limiter = limiter.memories(memories as usize);
    }
    store.store.data_mut().data.store_limits = limiter.build();
    store.store.limiter(|data| &mut data.data.store_limits);
}

#[repr(C)]
#[derive(Clone)]
pub struct wasmtime_component_val_record_field_t {
    pub name: wasm_name_t,
    pub val: wasmtime_component_val_t,
}

impl Default for wasmtime_component_val_record_field_t {
    fn default() -> Self {
        Self {
            name: Vec::new().into(),
            val: Default::default(),
        }
    }
}

declare_vecs! {
    (
        name: wasmtime_component_val_vec_t,
        ty: wasmtime_component_val_t,
        new: wasmtime_component_val_vec_new,
        empty: wasmtime_component_val_vec_new_empty,
        uninit: wasmtime_component_val_vec_new_uninitialized,
        copy: wasmtime_component_val_vec_copy,
        delete: wasmtime_component_val_vec_delete,
    )
    (
        name: wasmtime_component_val_record_field_vec_t,
        ty: wasmtime_component_val_record_field_t,
        new: wasmtime_component_val_record_field_vec_new,
        empty: wasmtime_component_val_record_field_vec_new_empty,
        uninit: wasmtime_component_val_record_field_vec_new_uninitialized,
        copy: wasmtime_component_val_record_field_vec_copy,
        delete: wasmtime_component_val_record_field_vec_delete,
    )
    (
        name: wasmtime_component_val_flags_vec_t,
        ty: wasm_name_t,
        new: wasmtime_component_val_flags_vec_new,
        empty: wasmtime_component_val_flags_vec_new_empty,
        uninit: wasmtime_component_val_flags_vec_new_uninitialized,
        copy: wasmtime_component_val_flags_vec_copy,
        delete: wasmtime_component_val_flags_vec_delete,
    )
}

#[repr(C)]
#[derive(Clone)]
pub struct wasmtime_component_val_variant_t {
    pub name: wasm_name_t,
    pub val: Option<Box<wasmtime_component_val_t>>,
}

#[repr(C)]
#[derive(Clone)]
pub struct wasmtime_component_val_result_t {
    pub value: Option<Box<wasmtime_component_val_t>>,
    pub error: bool,
}

#[repr(C)]
#[derive(Clone)]
pub struct wasmtime_component_val_enum_t {
    pub name: wasm_name_t,
}

// todo
// #[no_mangle]
// pub extern "C" fn wasmtime_component_val_flags_set(
//     flags: &mut wasmtime_component_val_flags_vec_t,
//     name: wasm_name_t,
//     enabled: bool,
// ) {
//     let mut f = flags.take();
//     let (idx, bit) = ((index / u32::BITS) as usize, index % u32::BITS);
//     if idx >= f.len() {
//         f.resize(idx + 1, Default::default());
//     }
//     if enabled {
//         f[idx] |= 1 << (bit);
//     } else {
//         f[idx] &= !(1 << (bit));
//     }
//     flags.set_buffer(f);
// }

// #[no_mangle]
// pub extern "C" fn wasmtime_component_val_flags_test(
//     flags: &wasmtime_component_val_flags_vec_t,
//     index: u32,
// ) -> bool {
//     let flags = flags.as_slice();
//     let (idx, bit) = ((index / u32::BITS) as usize, index % u32::BITS);
//     flags.get(idx).map(|v| v & (1 << bit) != 0).unwrap_or(false)
// }

#[repr(C, u8)]
#[derive(Clone)]
pub enum wasmtime_component_val_t {
    Bool(bool),
    S8(i8),
    U8(u8),
    S16(i16),
    U16(u16),
    S32(i32),
    U32(u32),
    S64(i64),
    U64(u64),
    F32(f32),
    F64(f64),
    Char(char),
    String(wasmtime_component_val_string_t),
    List(wasmtime_component_val_vec_t),
    Record(wasmtime_component_val_record_field_vec_t),
    Tuple(wasmtime_component_val_vec_t),
    Variant(wasmtime_component_val_variant_t),
    Enum(wasmtime_component_val_enum_t),
    Option(Option<Box<wasmtime_component_val_t>>),
    Result(wasmtime_component_val_result_t),
    Flags(wasmtime_component_val_flags_vec_t),
}

impl wasmtime_component_val_t {
    fn into_val(self) -> Result<Val> {
        Ok(match self {
            wasmtime_component_val_t::Bool(b) => Val::Bool(b),
            wasmtime_component_val_t::S8(v) => Val::S8(v),
            wasmtime_component_val_t::U8(v) => Val::U8(v),
            wasmtime_component_val_t::S16(v) => Val::S16(v),
            wasmtime_component_val_t::U16(v) => Val::U16(v),
            wasmtime_component_val_t::S32(v) => Val::S32(v),
            wasmtime_component_val_t::U32(v) => Val::U32(v),
            wasmtime_component_val_t::S64(v) => Val::S64(v),
            wasmtime_component_val_t::U64(v) => Val::U64(v),
            wasmtime_component_val_t::F32(v) => Val::Float32(v),
            wasmtime_component_val_t::F64(v) => Val::Float64(v),
            wasmtime_component_val_t::Char(v) => Val::Char(v),
            wasmtime_component_val_t::String(mut v) => Val::String(String::from_utf8(v.take())?),
            wasmtime_component_val_t::List(mut v) => {
                let v: Vec<_> =
                    v.take()
                        .into_iter()
                        .map(|v| v.into_val())
                        .collect::<Result<Vec<_>>>()?;
                Val::List(v)
            }
            wasmtime_component_val_t::Record(mut v) => {
                let v: Vec<_> = v.take().into_iter().map(|it| {
                    let name = (&it.name).try_into()?;
                    Ok((name, it.val.into_val()?))
                }).collect::<Result<Vec<_>>>()?;
                Val::Record(v)
            }
            wasmtime_component_val_t::Tuple(mut v) => {
                let v = v.take().into_iter()
                    .map(|v| v.into_val())
                    .collect::<Result<Vec<_>>>()?;
                Val::Tuple(v)
            }
            wasmtime_component_val_t::Variant(v) => {
                if let Some(val) = v.val {
                    Val::Variant((&v.name).try_into()?, Some(Box::new(val.into_val()?)))
                } else {
                    Val::Variant((&v.name).try_into()?, None)
                }
            }
            wasmtime_component_val_t::Enum(v) => {
                Val::Enum((&v.name).try_into()?)
            }
            wasmtime_component_val_t::Option(v) => {
                Val::Option({
                    match v {
                        Some(v) => Some(Box::new(v.into_val()?)),
                        None => None,
                    }
                })
            }
            wasmtime_component_val_t::Result(v) => {
                let v: Result<Option<Box<Val>>, Option<Box<Val>>> = if v.error {
                    Ok(match v.value {
                        Some(v) => Some(Box::new(v.into_val()?)),
                        None => None,
                    })
                } else {
                    Ok(match v.value {
                        Some(v) => Some(Box::new(v.into_val()?)),
                        None => None,
                    })
                };
                Val::Result(v)
            }
            wasmtime_component_val_t::Flags(flags) => {
                Val::Flags(flags.as_slice().iter().map(|it| Ok(it.try_into()?)).collect::<Result<Vec<String>>>()?)
            }
        })
    }
}

impl wasmtime_component_val_t {
    // type Error = anyhow::Error;

    fn try_from(value: &Val) -> Result<Self, anyhow::Error> {
        Ok(match value {
            Val::Bool(v) => wasmtime_component_val_t::Bool(*v),
            Val::S8(v) => wasmtime_component_val_t::S8(*v),
            Val::U8(v) => wasmtime_component_val_t::U8(*v),
            Val::S16(v) => wasmtime_component_val_t::S16(*v),
            Val::U16(v) => wasmtime_component_val_t::U16(*v),
            Val::S32(v) => wasmtime_component_val_t::S32(*v),
            Val::U32(v) => wasmtime_component_val_t::U32(*v),
            Val::S64(v) => wasmtime_component_val_t::S64(*v),
            Val::U64(v) => wasmtime_component_val_t::U64(*v),
            Val::Float32(v) => wasmtime_component_val_t::F32(*v),
            Val::Float64(v) => wasmtime_component_val_t::F64(*v),
            Val::Char(v) => wasmtime_component_val_t::Char(*v),
            Val::String(v) => {
                let v = v.to_string().into_bytes();
                wasmtime_component_val_t::String(v.into())
            }
            Val::List(v) => {
                let v = v.iter().map(|v| Self::try_from(v)).collect::<Result<Vec<_>>>()?;
                wasmtime_component_val_t::List(v.into())
            }
            Val::Record(v) => {
                let v = v.into_iter()
                    .map(|(name, v)| {
                        Ok(wasmtime_component_val_record_field_t {
                            name: name.to_string().into_bytes().into(),
                            val: Self::try_from(v)?,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                wasmtime_component_val_t::Record(v.into())
            }
            Val::Tuple(v) => {
                let v = v
                    .into_iter()
                    .map(|v| Self::try_from(v))
                    .collect::<Result<Vec<_>>>()?;
                wasmtime_component_val_t::Tuple(v.into())
            }
            Val::Variant(discriminant, v) => {
                let val = match v {
                    Some(v) => Some(Box::new(
                        Self::try_from(v)?)),
                    None => None,
                };
                wasmtime_component_val_t::Variant(wasmtime_component_val_variant_t {
                    name: wasm_name_t::from_name(discriminant.clone()),
                    val: val
                })
            }
            Val::Enum(v) => {
                wasmtime_component_val_t::Enum(wasmtime_component_val_enum_t {
                    name: wasm_name_t::from_name(v.clone()),
                })
            }
            Val::Option(v) => wasmtime_component_val_t::Option(match v {
                Some(v) => Some(Box::new(Self::try_from(v)?)),
                None => None,
            }),
            Val::Result(v) => {
                let (error, value) = match v {
                    Ok(v) => (false, v),
                    Err(v) => (true, v),
                };
                let value = match value {
                    Some(v) => Some(Box::new(Self::try_from(v)?)),
                    None => None,
                };
                wasmtime_component_val_t::Result(wasmtime_component_val_result_t { value, error })
            }
            Val::Flags(v) => {
                let flags = v.iter().map(|name| wasm_name_t::from_name(name.clone())).collect::<Vec<_>>().into();
                wasmtime_component_val_t::Flags(flags)
            }
            Val::Resource(_) => bail!("resource types are unimplemented"),
        })
    }
}

impl Default for wasmtime_component_val_t {
    fn default() -> Self {
        Self::Bool(false)
    }
}

pub type wasmtime_component_val_kind_t = u8;
pub const WASMTIME_COMPONENT_VAL_KIND_BOOL: wasmtime_component_val_kind_t = 0;
pub const WASMTIME_COMPONENT_VAL_KIND_S8: wasmtime_component_val_kind_t = 1;
pub const WASMTIME_COMPONENT_VAL_KIND_U8: wasmtime_component_val_kind_t = 2;
pub const WASMTIME_COMPONENT_VAL_KIND_S16: wasmtime_component_val_kind_t = 3;
pub const WASMTIME_COMPONENT_VAL_KIND_U16: wasmtime_component_val_kind_t = 4;
pub const WASMTIME_COMPONENT_VAL_KIND_S32: wasmtime_component_val_kind_t = 5;
pub const WASMTIME_COMPONENT_VAL_KIND_U32: wasmtime_component_val_kind_t = 6;
pub const WASMTIME_COMPONENT_VAL_KIND_S64: wasmtime_component_val_kind_t = 7;
pub const WASMTIME_COMPONENT_VAL_KIND_U64: wasmtime_component_val_kind_t = 8;
pub const WASMTIME_COMPONENT_VAL_KIND_FLOAT_32: wasmtime_component_val_kind_t = 9;
pub const WASMTIME_COMPONENT_VAL_KIND_FLOAT_64: wasmtime_component_val_kind_t = 10;
pub const WASMTIME_COMPONENT_VAL_KIND_CHAR: wasmtime_component_val_kind_t = 11;
pub const WASMTIME_COMPONENT_VAL_KIND_STRING: wasmtime_component_val_kind_t = 12;
pub const WASMTIME_COMPONENT_VAL_KIND_LIST: wasmtime_component_val_kind_t = 13;
pub const WASMTIME_COMPONENT_VAL_KIND_RECORD: wasmtime_component_val_kind_t = 14;
pub const WASMTIME_COMPONENT_VAL_KIND_TUPLE: wasmtime_component_val_kind_t = 15;
pub const WASMTIME_COMPONENT_VAL_KIND_VARIANT: wasmtime_component_val_kind_t = 16;
pub const WASMTIME_COMPONENT_VAL_KIND_ENUM: wasmtime_component_val_kind_t = 17;
pub const WASMTIME_COMPONENT_VAL_KIND_OPTION: wasmtime_component_val_kind_t = 18;
pub const WASMTIME_COMPONENT_VAL_KIND_RESULT: wasmtime_component_val_kind_t = 19;
pub const WASMTIME_COMPONENT_VAL_KIND_FLAGS: wasmtime_component_val_kind_t = 20;

#[repr(transparent)]
pub struct wasmtime_component_t {
    component: Component,
}

#[no_mangle]
pub unsafe extern "C" fn wasmtime_component_from_binary(
    engine: &wasm_engine_t,
    bytes: *const u8,
    len: usize,
    out: &mut *mut wasmtime_component_t,
) -> Option<Box<wasmtime_error_t>> {
    let bytes = crate::slice_from_raw_parts(bytes, len);
    handle_result(Component::from_binary(&engine.engine, bytes), |component| {
        *out = Box::into_raw(Box::new(wasmtime_component_t { component }));
    })
}

#[no_mangle]
pub unsafe extern "C" fn wasmtime_component_delete(_: Box<wasmtime_component_t>) {}

#[repr(transparent)]
pub struct wasmtime_component_linker_t {
    linker: Linker<Host>,
}

#[no_mangle]
pub extern "C" fn wasmtime_component_linker_new(
    engine: &wasm_engine_t,
) -> Box<wasmtime_component_linker_t> {
    Box::new(wasmtime_component_linker_t {
        linker: Linker::new(&engine.engine),
    })
}

#[no_mangle]
pub extern "C" fn wasmtime_component_linker_link_wasi(
    linker: &mut wasmtime_component_linker_t,
    trap_ret: &mut *mut wasm_trap_t,
) -> Option<Box<wasmtime_error_t>> {
    match wasmtime_wasi::add_to_linker_sync(&mut linker.linker) {
        Ok(_) => None,
        Err(e) => handle_call_error(e, trap_ret),
    }
}

pub type wasmtime_component_func_callback_t = extern "C" fn(
    *mut c_void,
    *const wasmtime_component_val_t,
    usize,
    *mut wasmtime_component_val_t,
    usize,
) -> Option<Box<wasm_trap_t>>;

unsafe fn c_callback_to_rust_fn<T>(
    callback: wasmtime_component_func_callback_t,
    data: *mut c_void,
    finalizer: Option<extern "C" fn(*mut std::ffi::c_void)>,
) -> impl Fn(StoreContextMut<'_, T>, &[Val], &mut [Val]) -> Result<()> + Send + Sync + 'static {
    let foreign = crate::ForeignData { data, finalizer };
    move |_store, params, results| {
        let _ = &foreign; // move entire foreign into this closure

        // Convert `params/results` to `wasmtime_component_val_t`. Use the previous
        // storage in `hostcall_val_storage` to help avoid allocations all the
        // time.

        // todo
        // let mut vals = mem::take(&mut caller.data_mut().hostcall_val_storage);
        // debug_assert!(vals.is_empty());
        let mut vals = vec![];

        vals.reserve(params.len() + results.len());
        vals.extend(
            params
                .iter()
                .map(|p| wasmtime_component_val_t::try_from(p).unwrap()),
        );
        vals.extend((0..results.len()).map(|_| wasmtime_component_val_t::default()));
        let (params, out_results) = vals.split_at_mut(params.len());

        // Invoke the C function pointer, getting the results.

        let out = callback(
            foreign.data,
            params.as_ptr(),
            params.len(),
            out_results.as_mut_ptr(),
            out_results.len(),
        );
        if let Some(trap) = out {
            return Err(trap.error);
        }

        // Translate the `wasmtime_component_val_t` results into the `results` space
        for (i, result) in out_results.iter().enumerate() {
            // todo: no clone
            results[i] = result.clone().into_val()?;
        }

        // todo
        // // Move our `vals` storage back into the store now that we no longer
        // // need it. This'll get picked up by the next hostcall and reuse our
        // // same storage.
        // vals.truncate(0);
        // caller.caller.data_mut().hostcall_val_storage = vals;
        Ok(())
    }
}

#[no_mangle]
pub unsafe extern "C" fn wasmtime_component_linker_func_new(
    linker: &mut wasmtime_component_linker_t,
    name: *const u8,
    len: usize,
    callback: wasmtime_component_func_callback_t,
    data: *mut c_void,
    finalizer: Option<extern "C" fn(*mut std::ffi::c_void)>,
) -> Option<Box<wasmtime_error_t>> {
    let name = crate::slice_from_raw_parts(name, len);
    let name = match std::str::from_utf8(name) {
        Ok(name) => name,
        Err(_) => return Some(Box::new(anyhow!("Invalid utf8").into())),
    };

    if let Some(mut instance) = linker.linker.root().get_instance("env") {
        return match instance.func_new(name, c_callback_to_rust_fn(callback, data, finalizer)) {
            Ok(_) => None,
            Err(e) => Some(Box::new(e.into())),
        };
    }

    let mut instance = linker.linker.instance("env").unwrap();

    match instance.func_new(name, c_callback_to_rust_fn(callback, data, finalizer)) {
        Ok(_) => None,
        Err(e) => Some(Box::new(e.into())),
    }
}

#[no_mangle]
pub extern "C" fn wasmtime_component_linker_instantiate(
    linker: &wasmtime_component_linker_t,
    store: WasmtimeStoreContextMutWasi<'_>,
    component: &wasmtime_component_t,
    out: &mut *mut wasmtime_component_instance_t,
    trap_ret: &mut *mut wasm_trap_t,
) -> Option<Box<wasmtime_error_t>> {
    // println!("wasmtime_component_linker_instantiate");
    match linker.linker.instantiate(store, &component.component) {
        Ok(instance) => {
            println!("wasmtime_component_linker_instantiate ok");
            *out = Box::into_raw(Box::new(wasmtime_component_instance_t { instance }));
            None
        }
        Err(e) => {
            // println!("wasmtime_component_linker_instantiate err {:?}", e);
            handle_call_error(e, trap_ret)
        }
    }
}

#[repr(transparent)]
pub struct wasmtime_component_instance_t {
    instance: Instance,
}

#[no_mangle]
pub unsafe extern "C" fn wasmtime_component_instance_get_func(
    instance: &wasmtime_component_instance_t,
    context: WasmtimeStoreContextMutWasi<'_>,
    name: *const u8,
    len: usize,
    item: &mut *mut wasmtime_component_func_t,
) -> bool {
    let name = crate::slice_from_raw_parts(name, len);
    let name = match std::str::from_utf8(name) {
        Ok(name) => name,
        Err(_) => return false,
    };
    let func = instance.instance.get_func(context, name);
    if let Some(func) = func {
        *item = Box::into_raw(Box::new(wasmtime_component_func_t { func }));
    }
    func.is_some()
}

#[repr(transparent)]
pub struct wasmtime_component_func_t {
    func: Func,
}

fn call_func(
    func: &wasmtime_component_func_t,
    mut context: WasmtimeStoreContextMutWasi<'_>,
    raw_params: &[wasmtime_component_val_t],
    raw_results: &mut [wasmtime_component_val_t],
) -> Result<()> {
    let params = raw_params.iter()
        .map(|v| v.clone().into_val())
        .collect::<Result<Vec<_>>>()?;
    let mut results = vec![Val::Bool(false); raw_results.len()];
    func.func.call(context.as_context_mut(), &params, &mut results)?;
    for (i, r) in results.iter().enumerate() {
        raw_results[i] = wasmtime_component_val_t::try_from(r)?;
    }
    func.func.post_return(context)?;
    Ok(())
}

#[no_mangle]
pub unsafe extern "C" fn wasmtime_component_func_call(
    func: &wasmtime_component_func_t,
    context: WasmtimeStoreContextMutWasi<'_>,
    params: *const wasmtime_component_val_t,
    params_len: usize,
    results: *mut wasmtime_component_val_t,
    results_len: usize,
    out_trap: &mut *mut wasm_trap_t,
) -> Option<Box<wasmtime_error_t>> {
    let raw_params = crate::slice_from_raw_parts(params, params_len);
    let mut raw_results = crate::slice_from_raw_parts_mut(results, results_len);
    match call_func(func, context, &raw_params, &mut raw_results) {
        Ok(_) => None,
        Err(e) => handle_call_error(e, out_trap),
    }
}

#[no_mangle]
pub unsafe extern "C" fn wasmtime_component_val_new() -> Box<wasmtime_component_val_t> {
    Box::new(wasmtime_component_val_t::default())
}

#[no_mangle]
pub unsafe extern "C" fn wasmtime_component_val_delete(val: *mut wasmtime_component_val_t) {
    _ = Box::from_raw(val)
}

#[cfg(test)]
mod tests {
    use crate::{
        wasmtime_component_val_flags_set, wasmtime_component_val_flags_vec_t,
        wasmtime_component_val_flags_test,
    };

    #[test]
    fn bit_fiddling() {
        let mut flags: wasmtime_component_val_flags_vec_t = Vec::new().into();
        wasmtime_component_val_flags_set(&mut flags, 1, true);
        assert!(wasmtime_component_val_flags_test(&flags, 1));
        assert!(!wasmtime_component_val_flags_test(&flags, 0));
        wasmtime_component_val_flags_set(&mut flags, 260, true);
        assert!(wasmtime_component_val_flags_test(&flags, 260));
        assert!(!wasmtime_component_val_flags_test(&flags, 261));
        assert!(!wasmtime_component_val_flags_test(&flags, 259));
        assert!(wasmtime_component_val_flags_test(&flags, 1));
        assert!(!wasmtime_component_val_flags_test(&flags, 0));
    }
}
