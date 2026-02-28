"""HyperTensor Billing — shadow metering and invoice generation.

Alpha scope:
    • Meter every successful job (CU = wall_time_s × device_multiplier)
    • Maintain an in-memory usage ledger
    • Generate shadow invoices (not charged, not sent to users)
"""
