# Generative AI (GenAI) on Kubernetes

Imagine being at a nice restaurant for dinner and the waiter hands you an encyclopedia-sized menu to choose from. This is how GenAI platform and workload owners feel when it comes to the never-ending list of tools in the AI ecosystem. This project provides an end-to-end workflow for training and running your own models.

---

## 1. Environment Setup

| Task | Tools | Notes |
|------|-------|-------|
| Provision GPU Nodes | `eksctl`, `GKE Autopilot`, kubeadm + GPU plugin | Use GPU-enabled instances like `g4dn`, `g6e`, or A100s |
| Install NVIDIA drivers (if needed) | [`nvidia-device-plugin`](https://github.com/NVIDIA/k8s-device-plugin) | DaemonSet to expose GPUs to Kubernetes |
| Package management | [`Helm`](https://helm.sh/) | Simplifies deploying components |

### 1.1 eksctl Example

Use [eksctl](https://eksctl.io/) to create a GPU-acceleratoed cluster:

```sh
$ eksctl create cluster -f - <<EOF
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig
metadata:
  name: $NAME
  region: $REGION
availabilityZones:
- us-west-2a
- us-west-2b
managedNodeGroups:
- name: ng-1
  desiredCapacity: $NUM_NODES
  # See https://aws.amazon.com/ec2/instance-types/ for other instance types.
  instanceType: g6e.xlarge
  ssh:
    allow: true
  availabilityZones:
  - us-west-2a
EOF
```

---

## 2. Training an LLM on Kubernetes

### 2.1 Data Preprocessing and Ingestion

| Task | Tools | Notes |
|------|-------|-------|
| Preprocess data | `Ray`, `Apache Spark`, `Kubeflow Pipelines` | Parallel preprocessing and tokenization |
| Data storage | `Ceph`, `S3`, `MinIO` | For storing training data |

### 2.2 Distributed Training

| Task | Tools | Notes |
|------|-------|-------|
| Define training job | `Kubeflow Training Operators`, `Volcano`, `Kueue`, `LWS` | GPU-aware scheduling; LWS is ideal for large, multi-GPU jobs |
| Frameworks | PyTorch, TensorFlow, JAX | Use DDP, NCCL, or Horovod |
| Launcher | `TorchElastic`, `Ray Train` | Elastic job orchestration |
| Monitor jobs | Prometheus + Grafana, TensorBoard | Real-time metrics and logs |

Example: Use a `PyTorchJob` CRD with 4 workers + 1 master using GPUs.

---

## 3. Checkpoints and Model Registry

| Task | Tools | Notes |
|------|-------|-------|
| Save checkpoints | `S3`, `GCS`, `PVC` | Store periodically during training |
| Model registry | `MLflow`, `Weights & Biases`, `Hugging Face Hub` | Track artifacts and metrics |

---

## 4. Model Evaluation / Fine-tuning

| Task | Tools | Notes |
|------|-------|-------|
| Fine-tuning | LoRA + Hugging Face Transformers + `vLLM` | Efficient low-rank adaptation |
| Hyperparam tuning | `Optuna`, `Ray Tune` | Automated optimization |

---

## 5. Containerize the Model Server

| Task | Tools | Notes |
|------|-------|-------|
| Build container | Docker, Podman | Include weights and model server |
| Base images | `vllm/vllm`, `text-generation-inference`, `tritonserver`, `tensorrt-llm` | Model-specific options |

---

## 6. Serve the LLM on Kubernetes

### 6.1 Inference Server

| Task | Tools | Notes |
|------|-------|-------|
| Model server | `vLLM`, `Triton`, `Text Generation Inference` | Optimized GPU serving |
| Batch + token streaming | `vLLM` | Ideal for OpenAI-compatible workloads |

### 6.2 Load Balancing and Autoscaling

| Task | Tools | Notes |
|------|-------|-------|
| Scheduling inference workloads | `LWS`, `Kueue`, `Volcano` | LWS can optimize placement for long-running GPU jobs |
| Auto Scaling | `KEDA`, HPA (custom metrics) | Trigger on GPU or queue depth |
| Routing | Kgateway, Envoy, `Gateway API Inference Extension` | Adapter-aware load balancing |

#### 6.2.1 Advanced Scheduling with LWS

[LWS (Lightweight Workload Scheduler)](https://lws.sigs.k8s.io) is a Kubernetes-native scheduler designed for latency-sensitive, GPU-intensive jobs like LLM training and inference. It supports:

- Job admission based on custom policies
- Co-scheduling for multi-GPU jobs
- Integration with Kueue, Volcano, and JobSets

LWS is particularly useful for:
- Long-running distributed training jobs (e.g., with NCCL or Tensor Parallelism)
- Managing backpressure when GPUs are oversubscribed
- Replacing custom queueing logic in complex workloads

### 6.4 Multi-Model Serving

| Task | Tools | Notes |
|------|-------|-------|
| Adapters & routing | LoRA adapters + `vLLM` + ConfigMaps | Hot-swap adapters |
| InferencePool resource | Gateway API + Kgateway + Envoy | Load balancing per model/load |

---

## 7. Monitoring and Observability

| Task | Tools | Notes |
|------|-------|-------|
| Metrics | Prometheus + Grafana, OpenTelemetry | QPS, latency, GPU usage |
| Logs | Loki, FluentBit, ELK | Log aggregation |
| Tracing | Jaeger, OpenTelemetry | Latency tracking |

---

## 8. Example End-to-End Stack

| Phase | Tool |
|-------|------|
| Training | Kubeflow + PyTorchJob + S3 |
| Serving | `vLLM` on GPU-backed Deployment |
| Ingress | Kgateway + Gateway API Inference Extension |
| Autoscaling | `KEDA` with GPU queue metrics |
| Monitoring | Prometheus + Grafana |

## Contributing

PRs are welcome to help maintain and improve this guide.
