@group(0) @binding(0)
var img: texture_storage_2d<r32float, read_write>;

@compute @workgroup_size(8, 8)
fn init(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(img);
     if (id.x >= dims.x || id.y >= dims.y) { return; }

        let uv = vec2<f32>(f32(id.x), f32(id.y)) / vec2<f32>(f32(dims.x), f32(dims.y));

        textureStore(img, id.xy, vec4<f32>(uv.x, 0, 0, 1));
}
