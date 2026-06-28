use crate::model::ModelId;
use crate::pipeline::{RunInput, RunOutput};
use serde::{Deserialize, Serialize};

/// Messages sent from the UI to the inference worker.
#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "kind")]
pub enum WorkerRequest {
    Load { model_id: ModelId },
    Run { request_id: usize, input: RunInput },
    Cancel { request_id: usize },
}

/// Messages sent from the inference worker to the UI.
#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "kind")]
pub enum WorkerEvent {
    Loading {
        model_id: ModelId,
        message: String,
    },
    Ready {
        model_id: ModelId,
    },
    Progress {
        request_id: usize,
        stage: String,
        current: usize,
        total: usize,
    },
    Completed {
        request_id: usize,
        output: RunOutput,
    },
    Failed {
        model_id: ModelId,
        message: String,
    },
    RunFailed {
        request_id: usize,
        message: String,
    },
    Cancelled {
        request_id: usize,
    },
}
