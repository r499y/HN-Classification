"""Sampling helpers delegating to PatientBagsDataset implementation extracted from notebooks."""


def select_indices_with_rng(dataset, X_full, idx_pool, rng, m_target):
    return dataset._select_indices_with_rng(X_full, idx_pool, rng, m_target)


def fixed_val_indices(dataset, pid, X, rng, m_target):
    return dataset._fixed_val_indices(pid, X, rng, m_target)


def rotating_train_indices(dataset, pid, X, rng, m_target):
    return dataset._rotating_train_indices(pid, X, rng, m_target)
