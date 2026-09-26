"""
Unit tests for Deep Learning Architecture & Physics-Informed Loss
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from src.models.encoders import WaveformEncoder
from src.models.fusion import CrossModalAttentionFusion, LatentHemodynamicModel
from src.models.physics_loss import PhysicsInformedLoss
from src.models.train import CompleteModelPipeline, apply_modality_dropout

def test_deep_learning_pipeline():
    batch_size = 4
    seq_len = 1250 # 10 seconds at 125 Hz
    horizon = 15
    
    # 1. Synthetic inputs (batch, 1, seq_len)
    ecg_tensor = torch.randn(batch_size, 1, seq_len)
    ppg_tensor = torch.randn(batch_size, 1, seq_len)
    target_trajectory = torch.full((batch_size, horizon), 85.0) + torch.randn(batch_size, horizon) * 5.0
    
    # 2. Encoders
    ecg_enc = WaveformEncoder(in_channels=1, hidden_dim=32, out_dim=64)
    ppg_enc = WaveformEncoder(in_channels=1, hidden_dim=32, out_dim=64)
    
    h_ecg = ecg_enc(ecg_tensor)
    h_ppg = ppg_enc(ppg_tensor)
    
    assert h_ecg.shape == (batch_size, 64, seq_len // 2), f"ECG shape mismatch: {h_ecg.shape}"
    assert h_ppg.shape == (batch_size, 64, seq_len // 2), f"PPG shape mismatch: {h_ppg.shape}"
    print(f"Encoders passed: output shape {h_ecg.shape}")
    
    # 3. Cross-modal fusion
    fusion_module = CrossModalAttentionFusion(feature_dim=64, num_heads=4)
    h_fused = fusion_module(h_ecg, h_ppg)
    assert h_fused.shape == (batch_size, 64, seq_len // 2)
    print("Cross-modal fusion passed")
    
    # 4. Latent Hemodynamic Model
    model = LatentHemodynamicModel(feature_dim=64, latent_dim=6, forecast_horizon=horizon)
    preds = model(h_fused)
    
    assert preds['z'].shape == (batch_size, 6)
    assert preds['forecast'].shape == (batch_size, horizon)
    assert preds['R'].shape == (batch_size, 1)
    assert preds['C'].shape == (batch_size, 1)
    
    # Strict physiological positivity check
    assert torch.all(preds['R'] > 0.0), "R must be strictly positive"
    assert torch.all(preds['C'] > 0.0), "C must be strictly positive"
    print(f"Model outputs passed: R range [{preds['R'].min().item():.2f}, {preds['R'].max().item():.2f}], C range [{preds['C'].min().item():.2f}, {preds['C'].max().item():.2f}]")
    
    # 5. Physics-Informed Loss & Backpropagation check
    loss_fn = PhysicsInformedLoss(w_obs=1.0, w_phys=0.2, w_param=0.05)
    loss_dict = loss_fn(preds, target_trajectory)
    
    total_loss = loss_dict['total_loss']
    assert not torch.isnan(total_loss), "Loss was NaN"
    
    # Backprop
    total_loss.backward()
    
    # Verify gradients in encoders, fusion, and heads
    assert ecg_enc.input_proj[0].weight.grad is not None, "ECG encoder received no gradients"
    assert model.param_head[0].weight.grad is not None, "Parameter head received no gradients"
    print(f"Backpropagation passed: Total Loss = {total_loss.item():.4f}, Forecast MAE = {loss_dict['forecast_mae'].item():.2f} mmHg")

def test_complete_pipeline_regime_3_and_4_inputs():
    batch_size = 2
    seq_len = 625
    horizon = 15

    model = CompleteModelPipeline(forecast_horizon=horizon)
    ecg = torch.randn(batch_size, 1, seq_len)
    ppg = torch.randn(batch_size, 1, seq_len)
    abp = torch.randn(batch_size, 1, seq_len) * 8.0 + 85.0
    intervention = torch.zeros(batch_size, 1, seq_len)
    intervention[:, :, seq_len // 2:] = 1.0

    regime3 = model(ecg, ppg, abp=abp, regime='ppg_ecg_abp')
    regime4 = model(ecg, ppg, abp=abp, intervention=intervention, regime='ppg_ecg_abp_intervention')

    assert regime3['forecast'].shape == (batch_size, horizon)
    assert regime4['forecast'].shape == (batch_size, horizon)
    assert regime4['u_summary'].shape == (batch_size, 1)
    assert torch.all(regime3['R'] > 0.0)
    assert torch.all(regime4['C'] > 0.0)
    print("Complete pipeline Regime 3/4 inputs passed")

def test_modality_dropout_keeps_ppg_and_masks_optional_modalities():
    batch_size = 3
    seq_len = 128
    ecg = torch.ones(batch_size, 1, seq_len)
    ppg = torch.ones(batch_size, 1, seq_len)
    abp = torch.ones(batch_size, 1, seq_len)
    intervention = torch.ones(batch_size, 1, seq_len)

    ecg_d, ppg_d, abp_d, intervention_d = apply_modality_dropout(
        ecg,
        ppg,
        abp=abp,
        intervention=intervention,
        dropout_probs={'ecg': 1.0, 'abp': 1.0, 'intervention': 1.0},
    )

    assert torch.all(ecg_d == 0.0)
    assert torch.all(ppg_d == 1.0)
    assert torch.all(abp_d == 0.0)
    assert torch.all(intervention_d == 0.0)
    print("Modality dropout behavior passed")

if __name__ == '__main__':
    test_deep_learning_pipeline()
    test_complete_pipeline_regime_3_and_4_inputs()
    test_modality_dropout_keeps_ppg_and_masks_optional_modalities()
    print("ALL DEEP LEARNING & PHYSICS TESTS PASSED!")
