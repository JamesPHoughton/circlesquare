# circlesquare
Agent Modeling of Software Vulnerability Discovery

This repository catalogs models and analysis scripts for simulating the discovery of cyber vulnerabilities.
The model itself is designed as the simplest representation of the discovery process that treats both vulnerabilities
and their discoverers at an individual level.


Todo:

Halfway through fitting the circles and squares model to the Massacci Nguyen data and performing their test. Our model takes a while to run, and is stochastic, and so optimization is difficult and slow, especially when we're trying to fit to multiple timeseries. There should be a way to cache the output of the model run and infer the parameters giving the best expectation of fitting the data curve. This isn't the same as what we would do to infer the true parameters of the model, however. 

Need to find a clever algorithm for doing parameter optimization to fit ~5000 curves with the same model, which produces stochastic output.
