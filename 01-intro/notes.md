# Notes Intro

## MLOps maturity models

The extent to which MLOps is implemented into a team or organization could be expressed as maturity. A framework for classifying different levels of MLOps maturity is listed below:

1. No MLOps
2. DevOps but no MLOps
3. Automated Training
4. Automated Model Deployment

### No MLOps

* ML process highly manual
* Poor cooperation between data scientists and software engineers
* Lack of standards, success depends on an individual's expertise

**Use cases**: Proof of concept (PoC), academic project

### DevOps but no MLOps

* ML training is most often manual
* Software engineers might help with the deployment
* Automated tests and releases

**Use cases**: Bringing PoC to production

### Automated Training

* ML experiment results are centrally tracked
* Training code and models are version controlled
* Deployment is handled by software engineers

**Use cases**: Maintaining 2-3+ ML models

### Automated Model Deployment

* ML models are deployed automatically
* Models are monitored for data drift and performance
* Models are retrained automatically

**Use cases**: Maintaining 10+ ML models

### Full MLOps Automated Operations

* Clearly defined metrics for model monitoring
* Automatic retraining triggered when passing a model metric's threshold

**Use cases**: Use only when a favorable trade-off between implementation cost and increase in efficiency is likely. Retraining is needed often and is repetitive (has potential for automation).

---

A high maturity level is not always needed because it comes with additional costs. The trade-off between automated model maintenance and the required effort to set up the automation should be considered. An ideal maturity level could be picked based on the use case / SLAs and the number of models deployed.

If you want to read more on maturity, visit [Microsoft's MLOps maturity model](https://docs.microsoft.com/en-us/azure/architecture/example-scenario/mlops/mlops-maturity-model).

## Questions

* Where does data versioning fit in?
