use crate::model::DEFAULT_MODEL_ID;
use crate::worker_protocol::{WorkerEvent, WorkerRequest};
use js_sys::{Array, JSON};
use wasm_bindgen::JsCast;
use wasm_bindgen::closure::Closure;
use web_sys::{Blob, BlobPropertyBag, MessageEvent, Url, Worker, WorkerOptions, WorkerType};

/// Browser module worker that runs Candle inference off the UI thread.
pub struct InferenceWorker {
    worker: Worker,
    script_url: String,
    _on_message: Closure<dyn FnMut(MessageEvent)>,
}

impl InferenceWorker {
    /// Creates the module worker and wires worker events into the supplied callback.
    pub fn new(mut on_event: impl FnMut(WorkerEvent) + 'static) -> Result<Self, String> {
        let (module_url, wasm_url) = find_trunk_wasm_assets()?;
        let (worker, script_url) = spawn_module_worker(&module_url, &wasm_url)?;
        let on_message = Closure::wrap(Box::new(move |event: MessageEvent| {
            let Some(message) = event.data().as_string() else {
                return;
            };
            match serde_json::from_str::<WorkerEvent>(&message) {
                Ok(event) => on_event(event),
                Err(error) => {
                    web_sys::console::error_1(&format!("Invalid worker event: {error}").into())
                }
            }
        }) as Box<dyn FnMut(MessageEvent)>);
        worker.set_onmessage(Some(on_message.as_ref().unchecked_ref()));

        Ok(Self {
            worker,
            script_url,
            _on_message: on_message,
        })
    }

    /// Posts one serialized request to the worker.
    #[inline]
    pub fn post(&self, request: &WorkerRequest) -> Result<(), String> {
        let message = serde_json::to_string(request).map_err(|error| error.to_string())?;
        let value = JSON::parse(&message).map_err(|error| format!("{error:?}"))?;
        self.worker
            .post_message(&value)
            .map_err(|error| format!("{error:?}"))
    }
}

impl Drop for InferenceWorker {
    fn drop(&mut self) {
        self.worker.terminate();
        let _ = Url::revoke_object_url(&self.script_url);
    }
}

fn find_trunk_wasm_assets() -> Result<(String, String), String> {
    let window = web_sys::window().ok_or_else(|| "missing browser window".to_string())?;
    let document = window
        .document()
        .ok_or_else(|| "missing browser document".to_string())?;
    let base_url = window
        .location()
        .href()
        .map_err(|error| format!("failed to get window location: {error:?}"))?;

    let script = document
        .query_selector("script[type='module']")
        .map_err(|error| format!("{error:?}"))?
        .ok_or_else(|| "missing Trunk module script".to_string())?
        .inner_html();
    let (module_url_raw, wasm_url_raw) = parse_trunk_wasm_assets(&script)?;

    let module_url = Url::new_with_base(module_url_raw, &base_url)
        .map_err(|error| format!("failed to resolve module URL: {error:?}"))?
        .href();
    let wasm_url = Url::new_with_base(wasm_url_raw, &base_url)
        .map_err(|error| format!("failed to resolve WASM URL: {error:?}"))?
        .href();

    Ok((module_url, wasm_url))
}

fn spawn_module_worker(module_url: &str, wasm_url: &str) -> Result<(Worker, String), String> {
    let default_model_id = DEFAULT_MODEL_ID.as_str();
    let script = format!(
        "const reportFailure = (stage, error) => {{\n\
           const message = error && error.message ? error.message : String(error);\n\
           self.postMessage(JSON.stringify({{ kind: 'Failed', model_id: {default_model_id:?}, message: `${{stage}} failed: ${{message}}` }}));\n\
         }};\n\
         self.onerror = (event) => {{\n\
           reportFailure('Worker runtime', event.error || event.message);\n\
         }};\n\
         self.onunhandledrejection = (event) => {{\n\
           reportFailure('Worker promise', event.reason);\n\
         }};\n\
         const queued = [];\n\
         let handle = null;\n\
         self.onmessage = (event) => {{\n\
           if (!handle) {{ queued.push(event.data); return; }}\n\
           try {{ handle(event.data); }} catch (error) {{ reportFailure('Worker message dispatch', error); }}\n\
         }};\n\
         try {{\n\
           self.__MOEBIUS_BASE_URL = new URL('./', {module_url:?}).href;\n\
           const module = await import({module_url:?});\n\
           await module.default({wasm_url:?});\n\
           handle = (data) => module.worker_handle_message(JSON.stringify(data));\n\
         }} catch (error) {{\n\
           reportFailure('Worker bootstrap', error);\n\
         }}\n\
         if (!handle) {{ queued.splice(0); }}\n\
         for (const data of queued.splice(0)) {{\n\
           try {{ handle(data); }} catch (error) {{ reportFailure('Worker queued message dispatch', error); }}\n\
         }}\n",
    );
    let parts = Array::new();
    parts.push(&script.into());

    let bag = BlobPropertyBag::new();
    bag.set_type("text/javascript");
    let blob = Blob::new_with_str_sequence_and_options(&parts, &bag)
        .map_err(|error| format!("{error:?}"))?;
    let url = Url::create_object_url_with_blob(&blob).map_err(|error| format!("{error:?}"))?;

    let options = WorkerOptions::new();
    options.set_type(WorkerType::Module);
    let worker = Worker::new_with_options(&url, &options).map_err(|error| format!("{error:?}"))?;
    Ok((worker, url))
}

fn parse_trunk_wasm_assets(script: &str) -> Result<(&str, &str), String> {
    let module_url = script_urls(script)
        .find(|url| url.ends_with(".js"))
        .ok_or_else(|| "missing Trunk JS module URL".to_string())?;
    let wasm_url = script_urls(script)
        .find(|url| url.ends_with(".wasm"))
        .ok_or_else(|| "missing Trunk WASM URL".to_string())?;

    Ok((module_url, wasm_url))
}

#[inline]
fn script_urls(script: &str) -> impl Iterator<Item = &str> {
    script.split(['"', '\'']).filter(is_asset_url)
}

#[inline]
fn is_asset_url(value: &&str) -> bool {
    value.starts_with('/') || value.starts_with("./") || value.starts_with("../")
}

#[cfg(test)]
mod tests {
    use super::parse_trunk_wasm_assets;

    #[test]
    fn parses_current_trunk_boot_script() {
        let script = "import init from '/moebius_app-abc.js'; init('/moebius_app-abc_bg.wasm');";

        let (module_url, wasm_url) = parse_trunk_wasm_assets(script).expect("script has assets");

        assert_eq!(module_url, "/moebius_app-abc.js");
        assert_eq!(wasm_url, "/moebius_app-abc_bg.wasm");
    }
}
