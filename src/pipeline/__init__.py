"""Pipeline for auditing and repairing the poisoned streamflow MLP.

Deliberately independent of src/vendor/streamflow_model.py: that module is the
correctness oracle, and a gate comparing a function to itself proves nothing.
See tests/pipeline/test_parity.py for the comparison.
"""
