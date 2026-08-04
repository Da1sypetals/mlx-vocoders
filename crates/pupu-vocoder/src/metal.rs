use std::ffi::{CStr, c_char};

use anyhow::{Result, bail, ensure};
use mlx_rs::{Array, Stream};

const OUTPUT_NAME: &CStr = c"out";
const EMPTY: &CStr = c"";

struct MetalKernel {
    kernel: mlx_sys::mlx_fast_metal_kernel,
}

unsafe impl Send for MetalKernel {}

impl MetalKernel {
    fn new(name: &CStr, input_names: &[&CStr], source: &CStr) -> Result<Self> {
        let mut input_name_values = input_names
            .iter()
            .map(|name| name.as_ptr())
            .collect::<Vec<*const c_char>>();
        let mut output_name_values = [OUTPUT_NAME.as_ptr()];
        let c_input_names = unsafe {
            mlx_sys::mlx_vector_string_new_data(
                input_name_values.as_mut_ptr(),
                input_name_values.len(),
            )
        };
        let c_output_names = unsafe {
            mlx_sys::mlx_vector_string_new_data(
                output_name_values.as_mut_ptr(),
                output_name_values.len(),
            )
        };
        let kernel = unsafe {
            mlx_sys::mlx_fast_metal_kernel_new(
                name.as_ptr(),
                c_input_names,
                c_output_names,
                source.as_ptr(),
                EMPTY.as_ptr(),
                true,
                false,
            )
        };
        unsafe {
            mlx_sys::mlx_vector_string_free(c_input_names);
            mlx_sys::mlx_vector_string_free(c_output_names);
        }
        ensure!(!kernel.ctx.is_null(), "创建 Metal kernel 失败");
        Ok(Self { kernel })
    }

    fn apply(
        &self,
        inputs: &[&Array],
        output_shape: &[i32],
        grid: [i32; 3],
        thread_group: [i32; 3],
        template_args: &[(&CStr, i32)],
    ) -> Result<Array> {
        let input_values = inputs
            .iter()
            .map(|input| input.as_ptr())
            .collect::<Vec<_>>();
        let c_inputs = unsafe {
            mlx_sys::mlx_vector_array_new_data(input_values.as_ptr(), input_values.len())
        };
        let mut c_outputs = unsafe { mlx_sys::mlx_vector_array_new() };
        let config = unsafe { mlx_sys::mlx_fast_metal_kernel_config_new() };

        let mut status = unsafe {
            mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
                config,
                output_shape.as_ptr(),
                output_shape.len(),
                mlx_sys::mlx_dtype__MLX_FLOAT32,
            )
        };
        status |= unsafe {
            mlx_sys::mlx_fast_metal_kernel_config_set_grid(config, grid[0], grid[1], grid[2])
        };
        status |= unsafe {
            mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(
                config,
                thread_group[0],
                thread_group[1],
                thread_group[2],
            )
        };
        for (name, value) in template_args {
            status |= unsafe {
                mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
                    config,
                    name.as_ptr(),
                    *value,
                )
            };
        }

        if status == 0 {
            let stream = Stream::task_local_or_default();
            status = unsafe {
                mlx_sys::mlx_fast_metal_kernel_apply(
                    &mut c_outputs,
                    self.kernel,
                    c_inputs,
                    config,
                    stream.as_ptr(),
                )
            };
        }
        let mut output = unsafe { mlx_sys::mlx_array_new() };
        if status == 0 {
            status = unsafe { mlx_sys::mlx_vector_array_get(&mut output, c_outputs, 0) };
        }
        unsafe {
            mlx_sys::mlx_fast_metal_kernel_config_free(config);
            mlx_sys::mlx_vector_array_free(c_inputs);
            mlx_sys::mlx_vector_array_free(c_outputs);
        }
        if status != 0 {
            unsafe { mlx_sys::mlx_array_free(output) };
            bail!("执行 Metal kernel 失败");
        }
        Ok(unsafe { Array::from_ptr(output) })
    }
}

impl Drop for MetalKernel {
    fn drop(&mut self) {
        unsafe { mlx_sys::mlx_fast_metal_kernel_free(self.kernel) }
    }
}

const ACTIVATION_TILE_SIZE: i32 = ACTIVATION_TILE * 2 + 11;
const ACTIVATION_TILE: i32 = 128;
const ACTIVATION_SOURCE: &CStr = c"
    uint lane = thread_index_in_threadgroup;
    uint block = threadgroup_position_in_grid.x;
    uint channel = threadgroup_position_in_grid.y;
    uint batch = threadgroup_position_in_grid.z;
    uint time = block * TILE_SIZE + lane;
    float alpha_value = metal::exp(alpha[channel]);
    float beta_value = metal::exp(beta[channel]);
    threadgroup float activation_tile[ACTIVATION_TILE_SIZE];

    for (uint tile_index = lane; tile_index < ACTIVATION_TILE_SIZE; tile_index += TILE_SIZE) {
        int high_time = int(block * TILE_SIZE * 2 + tile_index) - 5;
        high_time = metal::clamp(high_time, 0, LENGTH * 2 - 1);
        float current = 0.0f;
        float previous = 0.0f;
        int current_raw = 15 + high_time;
        int current_parity = current_raw & 1;
        int current_base = (current_raw - current_parity) / 2;

        for (int index = 0; index < 6; ++index) {
            int tap = current_parity + index * 2;
            int input_time = metal::clamp(current_base - index - 5, 0, LENGTH - 1);
            current += input[(batch * LENGTH + input_time) * CHANNELS + channel] * up_filter[tap];
        }
        if (high_time > 0) {
            int previous_raw = current_raw - 1;
            int previous_parity = previous_raw & 1;
            int previous_base = (previous_raw - previous_parity) / 2;
            for (int index = 0; index < 6; ++index) {
                int tap = previous_parity + index * 2;
                int input_time = metal::clamp(previous_base - index - 5, 0, LENGTH - 1);
                previous += input[(batch * LENGTH + input_time) * CHANNELS + channel] * up_filter[tap];
            }
        }

        current *= 2.0f;
        previous *= 2.0f;
        float delta = current - previous;
        float sum = current + previous;
        float sinc_argument = alpha_value * delta / M_PI_F;
        float sinc_phase = sinc_argument * M_PI_F;
        float sinc = sinc_phase == 0.0f ? 1.0f : metal::sin(sinc_phase) / sinc_phase;
        float periodic = 1.0f - metal::cos(alpha_value * sum) * sinc;
        activation_tile[tile_index] = sum / 2.0f + periodic / ((beta_value + 1e-9f) * 2.0f);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (time < LENGTH) {
        float value = 0.0f;
        for (uint tap = 0; tap < 12; ++tap) {
            value += activation_tile[lane * 2 + tap] * down_filter[tap];
        }
        out[(batch * LENGTH + time) * CHANNELS + channel] = value;
    }
";

pub struct MetalActivationKernel {
    kernel: MetalKernel,
}

impl MetalActivationKernel {
    pub fn new() -> Result<Self> {
        Ok(Self {
            kernel: MetalKernel::new(
                c"pupu_activation_1d",
                &[c"input", c"alpha", c"beta", c"up_filter", c"down_filter"],
                ACTIVATION_SOURCE,
            )?,
        })
    }

    pub fn forward(
        &self,
        input: &Array,
        alpha: &Array,
        beta: &Array,
        up_filter: &Array,
        down_filter: &Array,
    ) -> Result<Array> {
        let shape = input.shape();
        ensure!(shape.len() == 3, "Activation1d 输入必须是三维数组");
        let batch = shape[0];
        let length = shape[1];
        let channels = shape[2];
        ensure!(
            length > 0 && channels > 0,
            "Activation1d 输入维度必须大于零"
        );
        let blocks = (length + ACTIVATION_TILE - 1) / ACTIVATION_TILE;
        self.kernel.apply(
            &[input, alpha, beta, up_filter, down_filter],
            shape,
            [blocks * ACTIVATION_TILE, channels, batch],
            [ACTIVATION_TILE, 1, 1],
            &[
                (c"TILE_SIZE", ACTIVATION_TILE),
                (c"ACTIVATION_TILE_SIZE", ACTIVATION_TILE_SIZE),
                (c"LENGTH", length),
                (c"CHANNELS", channels),
            ],
        )
    }
}

const RESAMPLE_SOURCE: &CStr = c"
    uint element = thread_position_in_grid.x;
    uint channel = element % CHANNELS;
    uint position = element / CHANNELS;
    uint time = position % OUTPUT_LENGTH;
    uint batch = position / OUTPUT_LENGTH;
    float input_lowpass = 0.0f;
    float source_lowpass = 0.0f;

    for (int tap = 0; tap < FILTER_LENGTH; ++tap) {
        int filtered_time = metal::clamp(int(time) + tap - FILTER_HALF, 0, OUTPUT_LENGTH - 1);
        float coefficient = filter[tap];
        if (filtered_time % SCALE_FACTOR == 0) {
            int input_time = filtered_time / SCALE_FACTOR;
            input_lowpass += input[(batch * INPUT_LENGTH + input_time) * CHANNELS + channel] * coefficient;
        }
        source_lowpass += source[(batch * OUTPUT_LENGTH + filtered_time) * CHANNELS + channel] * coefficient;
    }

    float source_value = source[(batch * OUTPUT_LENGTH + time) * CHANNELS + channel];
    out[element] = input_lowpass + source_value - source_lowpass;
";

pub struct MetalResampleKernel {
    kernel: MetalKernel,
}

impl MetalResampleKernel {
    pub fn new() -> Result<Self> {
        Ok(Self {
            kernel: MetalKernel::new(
                c"pupu_resample_combine",
                &[c"input", c"source", c"filter"],
                RESAMPLE_SOURCE,
            )?,
        })
    }

    pub fn forward(
        &self,
        input: &Array,
        source: &Array,
        filter: &Array,
        scale_factor: i32,
    ) -> Result<Array> {
        let input_shape = input.shape();
        let output_shape = source.shape();
        ensure!(
            input_shape.len() == 3 && output_shape.len() == 3,
            "resample 输入必须是三维数组"
        );
        ensure!(
            input_shape[0] == output_shape[0] && input_shape[2] == output_shape[2],
            "resample 的 batch 和 channel 必须一致",
        );
        ensure!(
            input_shape[1] * scale_factor == output_shape[1],
            "resample 输出长度与 scale factor 不匹配",
        );
        let filter_length = filter.shape()[1];
        let elements = output_shape.iter().product::<i32>();
        self.kernel.apply(
            &[input, source, filter],
            output_shape,
            [elements, 1, 1],
            [256, 1, 1],
            &[
                (c"INPUT_LENGTH", input_shape[1]),
                (c"OUTPUT_LENGTH", output_shape[1]),
                (c"CHANNELS", output_shape[2]),
                (c"SCALE_FACTOR", scale_factor),
                (c"FILTER_LENGTH", filter_length),
                (c"FILTER_HALF", (filter_length - 1) / 2),
            ],
        )
    }
}
