// Patch: ggml renamed the Vulkan-specific backend functions to generic device
// API in whisper.cpp >=1.7.4 / ggml >=0.10. The old ggml_backend_vk_* symbols
// no longer exist; we enumerate all registered devices and filter to GPU/IGPU.
use std::ffi::CStr;
use whisper_rs_sys::{
    ggml_backend_buffer_type_t, ggml_backend_dev_buffer_type, ggml_backend_dev_count,
    ggml_backend_dev_description, ggml_backend_dev_get, ggml_backend_dev_memory,
    ggml_backend_dev_type, ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_GPU,
    ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_IGPU,
};

#[derive(Debug, Clone)]
pub struct VKVram {
    pub free: usize,
    pub total: usize,
}

/// Human-readable device information
#[derive(Debug, Clone)]
pub struct VkDeviceInfo {
    pub id: i32,
    pub name: String,
    pub vram: VKVram,
    /// Buffer type to pass to `whisper::Backend::create_buffer`
    pub buf_type: ggml_backend_buffer_type_t,
}

/// Enumerate every physical GPU ggml can see.
///
/// Filters the global device registry to GPU and IGPU entries, which on Linux
/// with Vulkan enabled correspond to Vulkan-backed devices.
///
/// Note: integrated GPUs are returned *after* discrete ones,
/// mirroring ggml's C logic.
pub fn list_devices() -> Vec<VkDeviceInfo> {
    unsafe {
        let n = ggml_backend_dev_count();
        let mut devices = Vec::new();
        let mut vk_id: i32 = 0;

        for i in 0..n {
            let dev = ggml_backend_dev_get(i);
            let dev_type = ggml_backend_dev_type(dev);
            if dev_type != ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_GPU
                && dev_type != ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_IGPU
            {
                continue;
            }

            let desc_ptr = ggml_backend_dev_description(dev);
            let name = if desc_ptr.is_null() {
                String::new()
            } else {
                CStr::from_ptr(desc_ptr).to_string_lossy().into_owned()
            };

            let mut free = 0usize;
            let mut total_mem = 0usize;
            ggml_backend_dev_memory(dev, &mut free, &mut total_mem);

            devices.push(VkDeviceInfo {
                id: vk_id,
                name,
                vram: VKVram { free, total: total_mem },
                buf_type: ggml_backend_dev_buffer_type(dev),
            });
            vk_id += 1;
        }

        devices
    }
}

#[cfg(test)]
mod vulkan_tests {
    use super::*;

    #[test]
    fn enumerate_must_not_panic() {
        let _ = list_devices();
    }

    #[test]
    fn sane_device_info() {
        let gpus = list_devices();
        let mut seen = std::collections::HashSet::new();

        for dev in &gpus {
            assert!(seen.insert(dev.id), "duplicated id {}", dev.id);
            assert!(!dev.name.trim().is_empty(), "GPU {} has empty name", dev.id);
            assert!(
                dev.vram.total >= dev.vram.free,
                "GPU {} total < free",
                dev.id
            );
        }
    }
}
