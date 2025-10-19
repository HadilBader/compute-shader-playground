use bevy::DefaultPlugins;
use bevy::app::{App, Plugin, Startup};
use bevy::asset::{Assets, DirectAssetAccessExt, Handle, RenderAssetUsages};
use bevy::image::Image;
use bevy::math::{Vec2, Vec3};
use bevy::prelude::{
    Camera2d, Commands, FromWorld, IntoScheduleConfigs, Res, ResMut, Resource, Sprite, Transform,
    World, default,
};
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_graph::{NodeRunError, RenderGraph, RenderGraphContext, RenderLabel};
use bevy::render::render_resource::binding_types::texture_storage_2d;
use bevy::render::render_resource::{
    BindGroup, BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries, CachedComputePipelineId,
    ComputePassDescriptor, ComputePipelineDescriptor, Extent3d, PipelineCache, ShaderStages,
    StorageTextureAccess, TextureDimension, TextureFormat, TextureUsages,
};
use bevy::render::renderer::{RenderContext, RenderDevice};
use bevy::render::texture::GpuImage;
use bevy::render::{Render, RenderApp, RenderSet, render_graph};
use std::borrow::Cow;
use bevy::log::info;

const SHADER_PATH: &str = "shader.wgsl";
const DISPLAY_FACTOR: u32 = 4;

const SIZE: (u32, u32) = (1280 / DISPLAY_FACTOR, 720 / DISPLAY_FACTOR);

const WORKGROUP_SIZE: u32 = 8;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(ComputeShaderPlugin)
        .add_systems(Startup, setup)
        .run();
}
fn setup(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let mut image = Image::new_fill(
        Extent3d {
            width: SIZE.0,
            height: SIZE.1,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[255, 0, 0, 255],
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::RENDER_WORLD,
    );

    image.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;

    let image_handle = images.add(image.clone());
    
    commands.spawn((
        Sprite {
            image: image_handle.clone(),
            custom_size: Some(Vec2::new(SIZE.0 as f32, SIZE.1 as f32)),
            ..default()
        },
        Transform::from_scale(Vec3::splat(DISPLAY_FACTOR as f32)),
    ));

    commands.spawn(Camera2d);
    commands.insert_resource(ComputeShaderImage {
        texture: image_handle,
    });
}

#[derive(Resource, Clone, ExtractResource)]
struct ComputeShaderImage {
    texture: Handle<Image>,
}

fn prepare_bind_group(
    mut commands: Commands,
    pipeline: Res<ComputeShaderPipeline>,
    gpu_image: Res<RenderAssets<GpuImage>>,
    compute_shader_image: Res<ComputeShaderImage>,
    render_device: Res<RenderDevice>,
) {
    let view = gpu_image.get(&compute_shader_image.texture).unwrap();
    let bind_group = render_device.create_bind_group(
        None,
        &pipeline.bind_group_layout,
        &BindGroupEntries::sequential((&view.texture_view,)),
    );
    commands.insert_resource(ComputeShaderBindGroup(bind_group))
}

struct ComputeShaderPlugin;

impl Plugin for ComputeShaderPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractResourcePlugin::<ComputeShaderImage>::default());

        let render_app = app.sub_app_mut(RenderApp);
        render_app.add_systems(
            Render,
            prepare_bind_group.in_set(RenderSet::PrepareBindGroups),
        );

        let mut render_graph = render_app.world_mut().resource_mut::<RenderGraph>();
        render_graph.add_node(ComputeShaderLabel, ComputeShaderNode::default());
        render_graph.add_node_edge(ComputeShaderLabel, bevy::render::graph::CameraDriverLabel);
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<ComputeShaderPipeline>();
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct ComputeShaderLabel;
#[derive(Resource)]
struct ComputeShaderPipeline {
    bind_group_layout: BindGroupLayout,
    pipeline: CachedComputePipelineId,
}

impl FromWorld for ComputeShaderPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let bind_group_layout = render_device.create_bind_group_layout(
            "Image",
            &BindGroupLayoutEntries::single(
                ShaderStages::COMPUTE,
                texture_storage_2d(TextureFormat::Rgba8Unorm, StorageTextureAccess::ReadWrite),
            ),
        );

        let shader = world.load_asset(SHADER_PATH);
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: None,
            layout: vec![bind_group_layout.clone()],
            push_constant_ranges: Vec::new(),
            shader: shader.clone(),
            shader_defs: vec![],
            entry_point: Cow::from("init"),
            zero_initialize_workgroup_memory: false,
        });
        ComputeShaderPipeline {
            bind_group_layout,
            pipeline,
        }
    }
}

#[derive(Resource)]
struct ComputeShaderBindGroup(BindGroup);

struct ComputeShaderNode;

impl Default for ComputeShaderNode {
    fn default() -> Self {
        Self
    }
}
impl render_graph::Node for ComputeShaderNode {
    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let bind_group = &world.resource::<ComputeShaderBindGroup>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<ComputeShaderPipeline>();

        if let Some(cpipeline) = pipeline_cache.get_compute_pipeline(pipeline.pipeline) {
            let mut pass = render_context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor::default());

            pass.set_bind_group(0, &bind_group.0, &[]);
            pass.set_pipeline(cpipeline);
            pass.dispatch_workgroups(SIZE.0 / WORKGROUP_SIZE, SIZE.1 / WORKGROUP_SIZE, 1);
        }
        Ok(())
    }
}
