use crate::model::{DEFAULT_MODEL_ID, ModelId};
use crate::pipeline::MoebiusPipeline;
use crate::worker_protocol::{WorkerEvent, WorkerRequest};
use std::cell::RefCell;
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::spawn_local;

thread_local! {
    static WORKER: RefCell<WorkerState> = RefCell::new(WorkerState::default());
}

#[derive(Default)]
struct WorkerState {
    pipeline: Option<MoebiusPipeline>,
    loading_model: Option<ModelId>,
    current_model: ModelId,
    active_run: Option<usize>,
}

/// Entrypoint invoked by the JavaScript module worker shim.
#[wasm_bindgen]
pub fn worker_handle_message(message: String) {
    console_error_panic_hook::set_once();

    match serde_json::from_str::<WorkerRequest>(&message) {
        Ok(WorkerRequest::Load { model_id }) => {
            WORKER.with(|state| {
                let mut state = state.borrow_mut();
                state.pipeline = None;
                state.loading_model = Some(model_id);
                state.active_run = None;
            });
            spawn_local(load_model(model_id));
        }
        Ok(WorkerRequest::Run { request_id, input }) => {
            WORKER.with(|state| state.borrow_mut().active_run = Some(request_id));
            spawn_local(run_pipeline(request_id, input));
        }
        Ok(WorkerRequest::Cancel { request_id }) => {
            clear_run(request_id);
            post_event(&WorkerEvent::Cancelled { request_id });
        }
        Err(error) => post_event(&WorkerEvent::Failed {
            model_id: current_model(),
            message: format!("Invalid worker request: {error}"),
        }),
    }
}

async fn run_pipeline(request_id: usize, input: crate::pipeline::RunInput) {
    let Some(mut pipeline) = take_pipeline() else {
        clear_run(request_id);
        post_event(&WorkerEvent::RunFailed {
            request_id,
            message: "Load the model first".to_string(),
        });
        return;
    };

    let result = pipeline
        .run(input, |progress| {
            post_event(&WorkerEvent::Progress {
                request_id,
                stage: progress.stage.to_string(),
                current: progress.current,
                total: progress.total,
            });
        })
        .await
        .map_err(|error| error.to_string());

    restore_pipeline(pipeline);

    match result {
        Ok(output) if is_run_current(request_id) => {
            clear_run(request_id);
            post_event(&WorkerEvent::Completed { request_id, output });
        }
        Err(message) if is_run_current(request_id) => {
            clear_run(request_id);
            post_event(&WorkerEvent::RunFailed {
                request_id,
                message,
            });
        }
        _ => post_event(&WorkerEvent::Cancelled { request_id }),
    }
}

async fn load_model(model_id: ModelId) {
    let result = MoebiusPipeline::load(model_id, |message| {
        post_event(&WorkerEvent::Loading { model_id, message })
    })
    .await;

    match result {
        Ok(pipeline) if is_loading_current(model_id) => {
            WORKER.with(|state| {
                let mut state = state.borrow_mut();
                state.pipeline = Some(pipeline);
                state.current_model = model_id;
                state.loading_model = None;
            });
            post_event(&WorkerEvent::Ready { model_id });
        }
        Err(error) if is_loading_current(model_id) => {
            WORKER.with(|state| state.borrow_mut().loading_model = None);
            post_event(&WorkerEvent::Failed {
                model_id,
                message: error.to_string(),
            });
        }
        _ => {}
    }
}

#[inline]
fn is_loading_current(model_id: ModelId) -> bool {
    WORKER.with(|state| state.borrow().loading_model == Some(model_id))
}

#[inline]
fn current_model() -> ModelId {
    WORKER.with(|state| state.borrow().current_model)
}

#[inline]
fn take_pipeline() -> Option<MoebiusPipeline> {
    WORKER.with(|state| state.borrow_mut().pipeline.take())
}

#[inline]
fn restore_pipeline(pipeline: MoebiusPipeline) {
    WORKER.with(|state| state.borrow_mut().pipeline = Some(pipeline));
}

#[inline]
fn is_run_current(request_id: usize) -> bool {
    WORKER.with(|state| state.borrow().active_run == Some(request_id))
}

#[inline]
fn clear_run(request_id: usize) {
    WORKER.with(|state| {
        let mut state = state.borrow_mut();
        if state.active_run == Some(request_id) {
            state.active_run = None;
        }
    });
}

fn post_event(event: &WorkerEvent) {
    let message = match serde_json::to_string(event) {
        Ok(message) => message,
        Err(error) => format!(
            r#"{{"kind":"Failed","model_id":"{}","message":"Failed to serialize worker event: {}"}}"#,
            DEFAULT_MODEL_ID.as_str(),
            error
        ),
    };

    let global = js_sys::global();
    let post_message = js_sys::Reflect::get(&global, &JsValue::from_str("postMessage"))
        .ok()
        .and_then(|value| value.dyn_into::<js_sys::Function>().ok());
    if let Some(post_message) = post_message {
        let _ = post_message.call1(&global, &JsValue::from_str(&message));
    }
}
