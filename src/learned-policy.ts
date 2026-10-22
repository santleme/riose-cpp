import policyWeights from '../training/policy/policy_weights-v2.json';
import { chooseSyntheticPolicyAction, type CowPolicyObservation, type PolicyArtifact } from '../training/policy/infer';
export type { CowPolicyObservation } from '../training/policy/infer';

const artifact=policyWeights as PolicyArtifact;

/**
 * Inference adapter for the Kaggle-trained synthetic DQN. Inputs are normalized
 * game state, not measured bovine physiology; see training/policy/README.md.
 */
export function chooseLearnedGameAction(observation:CowPolicyObservation):'graze'|'drink'|'rest'|'seekShade'|'followHerd'{
  const action=chooseSyntheticPolicyAction(observation,artifact);
  if(action==='seek_shade')return'seekShade';
  if(action==='follow_herd')return'followHerd';
  if(action==='graze'||action==='drink'||action==='rest')return action;
  return'rest';
}
