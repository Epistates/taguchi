//! Main effects calculation for DOE analysis.
//!
//! Calculates the effect of each factor level on the response variable.

use ndarray::Array2;

use super::types::MainEffect;

/// Calculate main effects for each factor.
///
/// Main effects measure how changing a factor level affects the response
/// relative to the grand mean. Factors are ranked by their range (effect span).
///
/// # Arguments
/// * `array_data` - Orthogonal array matrix (runs × factors), values are level indices
/// * `run_averages` - Average response for each run
/// * `grand_mean` - Overall mean of all responses
///
/// # Returns
/// * Vector of MainEffect for each factor, ranked by importance
///
/// # Algorithm
/// For each factor:
/// 1. Group runs by their level for this factor
/// 2. Calculate mean response at each level
/// 3. Calculate effect = level_mean - grand_mean
/// 4. Calculate range = max(level_means) - min(level_means)
/// 5. Rank factors by range (descending)
pub fn calculate_main_effects(
    array_data: &Array2<u32>,
    run_averages: &[f64],
    grand_mean: f64,
) -> Vec<MainEffect> {
    let num_factors = array_data.ncols();
    let mut effects: Vec<MainEffect> = Vec::with_capacity(num_factors);

    for factor_idx in 0..num_factors {
        // Determine number of levels for this factor
        let column = array_data.column(factor_idx);
        let num_levels = column.iter().copied().max().map(|m| m as usize + 1).unwrap_or(0);

        if num_levels == 0 {
            continue;
        }

        // Sum responses for each level
        let mut level_sums: Vec<f64> = vec![0.0; num_levels];
        let mut level_counts: Vec<usize> = vec![0; num_levels];

        for (run_idx, &level) in column.iter().enumerate() {
            let level_idx = level as usize;
            if level_idx < num_levels && run_idx < run_averages.len() {
                level_sums[level_idx] += run_averages[run_idx];
                level_counts[level_idx] += 1;
            }
        }

        // Calculate means
        let level_means: Vec<f64> = level_sums
            .iter()
            .zip(level_counts.iter())
            .map(|(&sum, &count)| {
                if count > 0 {
                    sum / count as f64
                } else {
                    grand_mean // Use grand mean for empty levels
                }
            })
            .collect();

        // Calculate effects (deviation from grand mean)
        let level_effects: Vec<f64> = level_means.iter().map(|m| m - grand_mean).collect();

        // Calculate range (max - min)
        let min_mean = level_means
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let max_mean = level_means
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let range = max_mean - min_mean;

        effects.push(MainEffect {
            factor_index: factor_idx,
            level_means,
            level_effects,
            range,
            rank: 0, // Will be set after sorting
        });
    }

    // Rank factors by range (higher range = more important = lower rank)
    let mut ranges: Vec<(usize, f64)> = effects
        .iter()
        .enumerate()
        .map(|(i, e)| (i, e.range))
        .collect();
    ranges.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    for (rank, (idx, _)) in ranges.iter().enumerate() {
        effects[*idx].rank = rank + 1;
    }

    effects
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_main_effects_l9() {
        // L9 example from Ross (1996)
        let array_data: Array2<u32> = array![
            [0, 0, 0, 0],
            [0, 1, 1, 1],
            [0, 2, 2, 2],
            [1, 0, 1, 2],
            [1, 1, 2, 0],
            [1, 2, 0, 1],
            [2, 0, 2, 1],
            [2, 1, 0, 2],
            [2, 2, 1, 0],
        ];

        let responses = vec![85.0, 92.0, 78.0, 91.0, 88.0, 82.0, 89.0, 86.0, 94.0];
        let grand_mean = responses.iter().sum::<f64>() / responses.len() as f64;

        let effects = calculate_main_effects(&array_data, &responses, grand_mean);

        assert_eq!(effects.len(), 4);

        // Check that ranks are assigned
        for effect in &effects {
            assert!(effect.rank >= 1 && effect.rank <= 4);
        }

        // Check that ranks are unique
        let mut ranks: Vec<usize> = effects.iter().map(|e| e.rank).collect();
        ranks.sort();
        assert_eq!(ranks, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_main_effects_two_level() {
        // Simple 2-level, 3-factor design
        let array_data: Array2<u32> = array![
            [0, 0, 0],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ];

        let responses = vec![10.0, 20.0, 15.0, 25.0];
        let grand_mean = 17.5;

        let effects = calculate_main_effects(&array_data, &responses, grand_mean);

        assert_eq!(effects.len(), 3);

        // Factor 0: level 0 mean = (10+20)/2 = 15, level 1 mean = (15+25)/2 = 20
        assert!((effects[0].level_means[0] - 15.0).abs() < 1e-10);
        assert!((effects[0].level_means[1] - 20.0).abs() < 1e-10);
        assert!((effects[0].range - 5.0).abs() < 1e-10);

        // Factor 1: level 0 mean = (10+15)/2 = 12.5, level 1 mean = (20+25)/2 = 22.5
        assert!((effects[1].level_means[0] - 12.5).abs() < 1e-10);
        assert!((effects[1].level_means[1] - 22.5).abs() < 1e-10);
        assert!((effects[1].range - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_main_effects_ranking() {
        // Design where factor 1 has the largest effect
        let array_data: Array2<u32> = array![
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ];

        // Factor 0: effect = 1, Factor 1: effect = 10
        let responses = vec![10.0, 20.0, 11.0, 21.0];
        let grand_mean = 15.5;

        let effects = calculate_main_effects(&array_data, &responses, grand_mean);

        // Factor 1 should be rank 1 (largest range = 10)
        assert_eq!(effects[1].rank, 1);
        // Factor 0 should be rank 2 (range = 1)
        assert_eq!(effects[0].rank, 2);
    }

    #[test]
    fn test_main_effects_level_effects() {
        let array_data: Array2<u32> = array![
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ];

        let responses = vec![10.0, 20.0, 30.0, 40.0];
        let grand_mean = 25.0;

        let effects = calculate_main_effects(&array_data, &responses, grand_mean);

        // Level effects should sum to approximately zero for each factor
        for effect in &effects {
            let effect_sum: f64 = effect.level_effects.iter().sum();
            assert!(effect_sum.abs() < 1e-10);
        }
    }
}
