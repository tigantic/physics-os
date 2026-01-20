//! Integration tests for FluidElite ZK

use fluidelite_zk::circuit::config::CircuitConfig;
use fluidelite_zk::field::Q16;
use fluidelite_zk::mpo::MPO;
use fluidelite_zk::mps::MPS;
use fluidelite_zk::ops::{add_mps, apply_mpo, fluidelite_step, readout};

#[test]
fn test_full_inference_step() {
    let config = CircuitConfig::test();

    // Create context
    let context = MPS::new(config.num_sites, config.chi_max, config.phys_dim);

    // Create weights
    let w_hidden = MPO::identity(config.num_sites, config.phys_dim);
    let w_input = MPO::identity(config.num_sites, config.phys_dim);

    // Run inference step
    let new_context = fluidelite_step(&context, 42, &w_hidden, &w_input, config.chi_max);

    // Verify structure
    assert_eq!(new_context.num_sites, config.num_sites);
    assert_eq!(new_context.cores[0].chi_left, 1);
    assert_eq!(new_context.cores[config.num_sites - 1].chi_right, 1);
}

#[test]
fn test_multiple_inference_steps() {
    let config = CircuitConfig::test();

    let mut context = MPS::new(config.num_sites, config.chi_max, config.phys_dim);
    let w_hidden = MPO::identity(config.num_sites, config.phys_dim);
    let w_input = MPO::identity(config.num_sites, config.phys_dim);

    // Run 10 steps
    for token in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] {
        context = fluidelite_step(&context, token, &w_hidden, &w_input, config.chi_max);

        // Verify invariants
        assert_eq!(context.num_sites, config.num_sites);
        assert!(context.max_chi() <= config.chi_max);
    }
}

#[test]
fn test_readout_produces_logits() {
    let config = CircuitConfig::test();

    let context = MPS::new(config.num_sites, config.chi_max, config.phys_dim);

    // Create readout weights
    let feature_size = config.chi_max * config.chi_max; // Simplified
    let readout_weights: Vec<Q16> = (0..config.vocab_size * feature_size)
        .map(|i| Q16::from_f64((i as f64) * 0.01))
        .collect();

    let logits = readout(&context, &readout_weights, config.vocab_size);

    assert_eq!(logits.len(), config.vocab_size);
}

#[test]
fn test_constraint_count_reasonable() {
    let config = CircuitConfig::production();
    let constraints = config.estimate_constraints();

    // Should be around 131k
    assert!(constraints > 50_000, "Too few constraints");
    assert!(constraints < 500_000, "Too many constraints");

    println!("Production constraint count: {}", constraints);
}

#[test]
fn test_proof_time_estimate() {
    let config = CircuitConfig::production();
    let time_ms = config.estimate_proof_time_ms();

    // Should be under 50ms on RTX 4090
    assert!(time_ms < 100.0, "Proof time too slow");

    println!("Estimated proof time: {:.2} ms", time_ms);
}

#[test]
fn test_mpo_serialization_roundtrip() {
    let config = CircuitConfig::test();

    let mut rng = rand::thread_rng();
    let mpo = MPO::random(config.num_sites, 2, config.phys_dim, &mut rng);

    let bytes = mpo.to_bytes();
    let mpo2 = MPO::from_bytes(&bytes).expect("Deserialization failed");

    assert_eq!(mpo.num_sites, mpo2.num_sites);

    for i in 0..mpo.num_sites {
        assert_eq!(mpo.cores[i].data.len(), mpo2.cores[i].data.len());
        for j in 0..mpo.cores[i].data.len() {
            assert_eq!(mpo.cores[i].data[j], mpo2.cores[i].data[j]);
        }
    }
}

#[test]
fn test_fixed_point_precision() {
    // Test that fixed-point arithmetic maintains precision
    let a = Q16::from_f64(1.5);
    let b = Q16::from_f64(2.5);

    // Multiplication
    let product = a.mul(b);
    let expected = 3.75;
    let error = (product.to_f64() - expected).abs();
    assert!(error < 0.0001, "Fixed-point multiplication error: {}", error);

    // Accumulation
    let mut sum = Q16::zero();
    for _ in 0..1000 {
        sum = sum + Q16::from_f64(0.001);
    }
    let expected = 1.0;
    let error = (sum.to_f64() - expected).abs();
    assert!(error < 0.01, "Fixed-point accumulation error: {}", error);
}

#[test]
fn test_embedding_uniqueness() {
    let num_sites = 8;

    // Each token should produce a unique MPS
    let mps_0 = MPS::embed_token(0, num_sites);
    let mps_1 = MPS::embed_token(1, num_sites);
    let mps_255 = MPS::embed_token(255, num_sites);

    // Check that they differ
    // Token 0 should have all |0⟩ states
    assert_eq!(mps_0.cores[0].get(0, 0, 0), Q16::one());
    assert_eq!(mps_0.cores[0].get(0, 1, 0), Q16::zero());

    // Token 1 should differ in last bit
    assert_eq!(mps_1.cores[num_sites - 1].get(0, 1, 0), Q16::one());

    // Token 255 should have all |1⟩ states
    for i in 0..num_sites {
        assert_eq!(mps_255.cores[i].get(0, 1, 0), Q16::one());
    }
}

#[test]
fn test_bond_dimension_growth_and_truncation() {
    let config = CircuitConfig::test();

    let mut context = MPS::new(config.num_sites, 2, config.phys_dim); // Start small
    let w_hidden = MPO::identity(config.num_sites, config.phys_dim);
    let w_input = MPO::identity(config.num_sites, config.phys_dim);

    // Track bond dimensions over steps
    let mut max_chis = Vec::new();

    for token in 0..20 {
        // Don't truncate to see natural growth
        let h_term = apply_mpo(&context, &w_hidden);
        let x_term = apply_mpo(&MPS::embed_token(token, config.num_sites), &w_input);
        context = add_mps(&h_term, &x_term);

        max_chis.push(context.max_chi());

        // Now truncate
        context.truncate(config.chi_max);
    }

    println!("Bond dimension growth: {:?}", max_chis);

    // After truncation, should stay bounded
    assert!(context.max_chi() <= config.chi_max);
}
