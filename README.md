## Deep learning predicts onset acceleration of 38 age-associated diseases from blood and body composition biomarkers in the UK Biobank

This repository contains the code for the experiments in the onset acceleration paper, where we use machine learning to predict onset speed of age-associated diseases and analyse disease-disease/disease-biomarker relationships.

{{ define "main" }}
<div class="">
    <iframe id="inlineFrameGraph"
    title="Inline Frame"
    src="correlation_graph.html"
    class="graph frame"
    >
</iframe>
</div>

{{end}}

### Simple usage

The loss functions for training a PyTorch model with the Cox PH loss is defined in `nn_metrics.py`. Specifically `neg_cox_log_likelihood` defines the loss function and `compute_c_index` defines the evaluation metric.

These functions can be used with any PyTorch neural network model. 

### Code structure

The main directory contains model training files, `train_nn.py` is the high-level executable for training neural networks. 

UK Biobank data preparation files begin with `make_`; researchers require their own access to the raw data.

Post-hoc analysis and plots are computed by scripts in the `analyse` directory. 
