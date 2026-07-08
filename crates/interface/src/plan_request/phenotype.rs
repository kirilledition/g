//! Phenotype run and compute-group compilation.

use g_plan as plan;

#[must_use]
pub(super) fn build_phenotype_run_plans(phenotype_names: &[String]) -> Vec<plan::PhenotypeRunPlan> {
    phenotype_names
        .iter()
        .enumerate()
        .map(|(phenotype_index, phenotype_name)| {
            let output_index = u32::try_from(phenotype_index + 1).expect("phenotype count must fit in u32");
            plan::PhenotypeRunPlan {
                phenotype_index: output_index,
                phenotype_name: phenotype_name.clone(),
                output_directory_name: plan::build_phenotype_output_directory_name(output_index, phenotype_name),
            }
        })
        .collect()
}
