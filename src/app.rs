use crate::imaging::IMG;
use crate::model::DEFAULT_MODEL_ID;
use crate::pipeline::{RunInput, RunOptions};
use crate::worker_client::InferenceWorker;
use crate::worker_protocol::{WorkerEvent, WorkerRequest};
use leptos::html::{Canvas, Input};
use leptos::prelude::*;
use std::cell::{Cell, RefCell};
use std::rc::Rc;
use wasm_bindgen::Clamped;
use wasm_bindgen::JsCast;
use wasm_bindgen::closure::Closure;
use web_sys::{
    CanvasRenderingContext2d, HtmlCanvasElement, HtmlImageElement, HtmlInputElement, ImageData,
    PointerEvent, Url,
};

/// Root Leptos component for the browser inpainting tool.
#[component]
pub fn App() -> impl IntoView {
    let image_ref = NodeRef::<Canvas>::new();
    let mask_ref = NodeRef::<Canvas>::new();
    let result_ref = NodeRef::<Canvas>::new();
    let file_ref = NodeRef::<Input>::new();

    let (status, set_status) = signal("Loading worker".to_string());
    let (progress, set_progress) = signal(0.0_f64);
    let (has_image, set_has_image) = signal(false);
    let (running, set_running) = signal(false);
    let (steps, set_steps) = signal(20_usize);
    let (guidance, set_guidance) = signal(2.0_f32);
    let (seed, set_seed) = signal(42_u32);
    let (paste, set_paste) = signal(true);
    let (download_url, set_download_url) = signal(None::<String>);

    let worker = Rc::new(RefCell::new(None::<InferenceWorker>));
    let request_id = Rc::new(Cell::new(0_usize));

    Effect::new({
        let worker = worker.clone();
        let request_id = request_id.clone();
        move |_| {
            let inference_worker = InferenceWorker::new({
                let request_id = request_id.clone();
                move |event| {
                    handle_worker_event(
                        event,
                        result_ref,
                        request_id.clone(),
                        set_status,
                        set_progress,
                        set_running,
                        set_download_url,
                    );
                }
            });

            match inference_worker {
                Ok(inference_worker) => {
                    if let Err(error) = inference_worker.post(&WorkerRequest::Load {
                        model_id: DEFAULT_MODEL_ID,
                    }) {
                        set_status.set(format!("Worker load failed: {error}"));
                    }
                    worker.replace(Some(inference_worker));
                }
                Err(error) => set_status.set(format!("Worker failed: {error}")),
            }
        }
    });

    install_mask_painting(mask_ref, has_image);

    let load_sample = move |_| {
        if let (Some(image), Some(mask)) = (image_ref.get(), mask_ref.get()) {
            draw_sample(&image);
            clear_canvas(&mask);
            set_has_image.set(true);
            set_status.set("Sample image ready".to_string());
            set_download_url.set(None);
        }
    };

    let load_file = move |_| {
        let Some(input) = file_ref.get() else {
            return;
        };
        let Some(file) = input.files().and_then(|files| files.get(0)) else {
            return;
        };
        let Some(image) = image_ref.get() else {
            return;
        };
        let Some(mask) = mask_ref.get() else {
            return;
        };

        load_file_into_canvas(
            file,
            image,
            mask,
            set_has_image,
            set_status,
            set_download_url,
        );
    };

    let clear_mask = move |_| {
        if let Some(mask) = mask_ref.get() {
            clear_canvas(&mask);
        }
    };

    let run = move |_| {
        if running.get_untracked() || !has_image.get_untracked() {
            return;
        }
        let Some(image) = image_ref.get() else {
            return;
        };
        let Some(mask) = mask_ref.get() else {
            return;
        };
        let worker_ref = worker.borrow();
        let Some(worker) = worker_ref.as_ref() else {
            set_status.set("Worker is not ready".to_string());
            return;
        };

        let next_request_id = request_id.get() + 1;
        request_id.set(next_request_id);
        set_running.set(true);
        set_progress.set(1.0);
        set_status.set("Preparing image".to_string());

        let input = RunInput {
            image_rgba: canvas_rgba(&image),
            mask_rgba: canvas_rgba(&mask),
            options: RunOptions {
                steps: steps.get_untracked(),
                guidance: guidance.get_untracked(),
                seed: seed.get_untracked(),
                paste: paste.get_untracked(),
            },
        };

        if let Err(error) = worker.post(&WorkerRequest::Run {
            request_id: next_request_id,
            input,
        }) {
            set_running.set(false);
            set_status.set(format!("Run failed: {error}"));
        }
    };

    view! {
        <div class="shell">
            <aside class="sidebar">
                <h1 class="brand">"Moebius"</h1>
                <p class="sidebar-copy">"Local browser inpainting with a Candle pipeline boundary."</p>

                <div class="field">
                    <label for="file">"Image"</label>
                    <input node_ref=file_ref id="file" type="file" accept="image/*" on:change=load_file />
                    <button type="button" on:click=load_sample>"Sample"</button>
                </div>

                <div class="field">
                    <div class="field-row">
                        <label for="steps">"Steps"</label>
                        <strong>{move || steps.get()}</strong>
                    </div>
                    <input
                        id="steps"
                        type="range"
                        min="4"
                        max="30"
                        value="20"
                        on:input=move |event| set_steps.set(event_target_value(&event).parse().unwrap_or(20))
                    />
                </div>

                <div class="field">
                    <div class="field-row">
                        <label for="guidance">"Guidance"</label>
                        <strong>{move || format!("{:.1}", guidance.get())}</strong>
                    </div>
                    <input
                        id="guidance"
                        type="range"
                        min="1"
                        max="5"
                        step="0.1"
                        value="2"
                        on:input=move |event| set_guidance.set(event_target_value(&event).parse().unwrap_or(2.0))
                    />
                </div>

                <div class="field">
                    <label for="seed">"Seed"</label>
                    <input
                        id="seed"
                        type="number"
                        value="42"
                        on:input=move |event| set_seed.set(event_target_value(&event).parse().unwrap_or(42))
                    />
                </div>

                <div class="field-row field">
                    <label for="paste">"Paste back"</label>
                    <input
                        id="paste"
                        type="checkbox"
                        checked=true
                        on:change=move |event| {
                            let checked = event
                                .target()
                                .and_then(|target| target.dyn_into::<HtmlInputElement>().ok())
                                .map(|input| input.checked())
                                .unwrap_or(true);
                            set_paste.set(checked);
                        }
                    />
                </div>

                <div class="actions">
                    <button type="button" on:click=clear_mask disabled=move || !has_image.get() || running.get()>"Clear Mask"</button>
                    <button class="primary" type="button" on:click=run disabled=move || !has_image.get() || running.get()>"Run"</button>
                </div>
            </aside>

            <main class="workspace">
                <header class="topbar">
                    <div class="status">{move || status.get()}</div>
                    <div class="progress-group">
                        <div
                            class="progress"
                            class:running=move || running.get() && progress.get() <= 1.0
                            role="progressbar"
                            aria-valuemin="0"
                            aria-valuemax="100"
                            aria-valuenow=move || format!("{:.0}", progress.get())
                        >
                            <div class="bar" style=move || format!("width: {:.1}%", progress.get())></div>
                        </div>
                        <span class="progress-value">{move || format!("{:.0}%", progress.get())}</span>
                    </div>
                </header>

                <section class="canvases">
                    <div class="canvas-pane">
                        <div class="canvas-label">"Image + mask"</div>
                        <div class="canvas-stack">
                            <canvas node_ref=image_ref width=IMG.to_string() height=IMG.to_string()></canvas>
                            <canvas class="mask-layer" node_ref=mask_ref width=IMG.to_string() height=IMG.to_string()></canvas>
                        </div>
                    </div>

                    <div class="canvas-pane">
                        <div class="canvas-label">"Result"</div>
                        <div class="canvas-stack">
                            <canvas node_ref=result_ref width=IMG.to_string() height=IMG.to_string()></canvas>
                        </div>
                    </div>
                </section>

                <footer class="footer">
                    {move || download_url.get().map(|url| view! {
                        <a class="download" href=url download="moebius-inpaint.png">"Download PNG"</a>
                    })}
                </footer>
            </main>
        </div>
    }
}

fn handle_worker_event(
    event: WorkerEvent,
    result_ref: NodeRef<Canvas>,
    request_id: Rc<Cell<usize>>,
    set_status: WriteSignal<String>,
    set_progress: WriteSignal<f64>,
    set_running: WriteSignal<bool>,
    set_download_url: WriteSignal<Option<String>>,
) {
    match event {
        WorkerEvent::Loading { message, .. } => set_status.set(message),
        WorkerEvent::Ready { .. } => set_status.set("Model ready".to_string()),
        WorkerEvent::Progress {
            request_id: event_id,
            stage,
            current,
            total,
        } if request_id.get() == event_id => {
            set_status.set(stage);
            if total > 0 {
                set_progress.set((current as f64 / total as f64) * 100.0);
            }
        }
        WorkerEvent::Completed {
            request_id: event_id,
            output,
        } if request_id.get() == event_id => {
            if let Some(canvas) = result_ref.get() {
                put_rgba(&canvas, &output.rgba);
                set_download_url.set(canvas.to_data_url().ok());
            }
            set_running.set(false);
            set_progress.set(100.0);
            set_status.set(format!("Done in {:.1}s", output.elapsed_ms / 1000.0));
        }
        WorkerEvent::RunFailed {
            request_id: event_id,
            message,
        } if request_id.get() == event_id => {
            set_running.set(false);
            set_progress.set(0.0);
            set_status.set(format!("Run failed: {message}"));
        }
        WorkerEvent::Failed { message, .. } => {
            set_running.set(false);
            set_progress.set(0.0);
            set_status.set(format!("Worker failed: {message}"));
        }
        WorkerEvent::Cancelled {
            request_id: event_id,
        } if request_id.get() == event_id => {
            set_running.set(false);
            set_progress.set(0.0);
            set_status.set("Run cancelled".to_string());
        }
        _ => {}
    }
}

fn install_mask_painting(mask_ref: NodeRef<Canvas>, has_image: ReadSignal<bool>) {
    Effect::new(move |_| {
        let Some(mask) = mask_ref.get() else {
            return;
        };
        let painting = Rc::new(Cell::new(false));
        let brush = Rc::new(Cell::new(40.0_f64));

        let down = Closure::wrap(Box::new({
            let painting = painting.clone();
            let brush = brush.clone();
            let mask = mask.clone();
            move |event: PointerEvent| {
                if !has_image.get_untracked() {
                    return;
                }
                painting.set(true);
                let _ = mask.set_pointer_capture(event.pointer_id());
                paint_at(&mask, &event, brush.get());
            }
        }) as Box<dyn FnMut(PointerEvent)>);
        mask.set_onpointerdown(Some(down.as_ref().unchecked_ref()));
        down.forget();

        let move_handler = Closure::wrap(Box::new({
            let painting = painting.clone();
            let brush = brush.clone();
            let mask = mask.clone();
            move |event: PointerEvent| {
                if painting.get() {
                    paint_at(&mask, &event, brush.get());
                }
            }
        }) as Box<dyn FnMut(PointerEvent)>);
        mask.set_onpointermove(Some(move_handler.as_ref().unchecked_ref()));
        move_handler.forget();

        let up = Closure::wrap(Box::new(move |_event: PointerEvent| painting.set(false))
            as Box<dyn FnMut(PointerEvent)>);
        mask.set_onpointerup(Some(up.as_ref().unchecked_ref()));
        mask.set_onpointercancel(Some(up.as_ref().unchecked_ref()));
        up.forget();
    });
}

fn paint_at(canvas: &HtmlCanvasElement, event: &PointerEvent, brush: f64) {
    let rect = canvas.get_bounding_client_rect();
    let x = ((event.client_x() as f64 - rect.left()) / rect.width()) * IMG as f64;
    let y = ((event.client_y() as f64 - rect.top()) / rect.height()) * IMG as f64;
    let ctx = context(canvas);
    ctx.set_fill_style_str("rgba(110, 168, 254, 0.62)");
    ctx.begin_path();
    let _ = ctx.arc(x, y, brush / 2.0, 0.0, std::f64::consts::PI * 2.0);
    ctx.fill();
}

fn load_file_into_canvas(
    file: web_sys::File,
    image: HtmlCanvasElement,
    mask: HtmlCanvasElement,
    set_has_image: WriteSignal<bool>,
    set_status: WriteSignal<String>,
    set_download_url: WriteSignal<Option<String>>,
) {
    let Ok(url) = Url::create_object_url_with_blob(&file) else {
        set_status.set("Could not read image file".to_string());
        return;
    };
    let img = HtmlImageElement::new().expect("browser creates image elements");
    let onload = Closure::wrap(Box::new({
        let img = img.clone();
        let url = url.clone();
        move || {
            draw_image_fit(&image, &img);
            clear_canvas(&mask);
            set_has_image.set(true);
            set_status.set("Image ready".to_string());
            set_download_url.set(None);
            let _ = Url::revoke_object_url(&url);
        }
    }) as Box<dyn FnMut()>);
    img.set_onload(Some(onload.as_ref().unchecked_ref()));
    onload.forget();
    img.set_src(&url);
}

fn draw_sample(canvas: &HtmlCanvasElement) {
    let ctx = context(canvas);
    ctx.set_fill_style_str("#d8ece8");
    ctx.fill_rect(0.0, 0.0, IMG as f64, IMG as f64);
    ctx.set_fill_style_str("#f8faf7");
    ctx.fill_rect(0.0, 210.0, IMG as f64, 302.0);
    ctx.set_fill_style_str("#506d83");
    ctx.fill_rect(0.0, 430.0, IMG as f64, 82.0);
    ctx.set_fill_style_str("#31423b");
    ctx.fill_rect(118.0, 190.0, 276.0, 190.0);
    ctx.set_fill_style_str("#f0f4ea");
    ctx.fill_rect(154.0, 226.0, 72.0, 76.0);
    ctx.fill_rect(286.0, 226.0, 72.0, 76.0);
    ctx.set_fill_style_str("#7c3f36");
    ctx.begin_path();
    ctx.move_to(88.0, 196.0);
    ctx.line_to(256.0, 96.0);
    ctx.line_to(424.0, 196.0);
    ctx.close_path();
    ctx.fill();
}

fn draw_image_fit(canvas: &HtmlCanvasElement, image: &HtmlImageElement) {
    let ctx = context(canvas);
    ctx.set_fill_style_str("#000");
    ctx.fill_rect(0.0, 0.0, IMG as f64, IMG as f64);
    let src_w = image.natural_width().max(1) as f64;
    let src_h = image.natural_height().max(1) as f64;
    let scale = (IMG as f64 / src_w).min(IMG as f64 / src_h);
    let w = (src_w * scale).round();
    let h = (src_h * scale).round();
    let x = ((IMG as f64 - w) / 2.0).floor();
    let y = ((IMG as f64 - h) / 2.0).floor();
    let _ = ctx.draw_image_with_html_image_element_and_dw_and_dh(image, x, y, w, h);
}

fn clear_canvas(canvas: &HtmlCanvasElement) {
    context(canvas).clear_rect(0.0, 0.0, IMG as f64, IMG as f64);
}

fn canvas_rgba(canvas: &HtmlCanvasElement) -> Vec<u8> {
    context(canvas)
        .get_image_data(0.0, 0.0, IMG as f64, IMG as f64)
        .expect("canvas image data is readable")
        .data()
        .to_vec()
}

fn put_rgba(canvas: &HtmlCanvasElement, rgba: &[u8]) {
    let data = ImageData::new_with_u8_clamped_array_and_sh(Clamped(rgba), IMG as u32, IMG as u32)
        .expect("rgba bytes match canvas dimensions");
    let _ = context(canvas).put_image_data(&data, 0.0, 0.0);
}

#[inline]
fn context(canvas: &HtmlCanvasElement) -> CanvasRenderingContext2d {
    canvas
        .get_context("2d")
        .expect("2d context lookup succeeds")
        .expect("2d context is available")
        .dyn_into::<CanvasRenderingContext2d>()
        .expect("context is a 2d context")
}
