/** Dependency-free browser inference for the exported synthetic DQN policy. */
export interface PolicyArtifact {
  schema_version: 1;
  model_type: 'mlp_q_argmax';
  input_fields: string[];
  actions: string[];
  hidden_activation: 'relu';
  layers: Array<{ weights_out_in: number[][]; bias: number[] }>;
  training_scope: 'synthetic_game_dynamics_only';
}

export type CowPolicyObservation = [
  hunger: number, thirst: number, fatigue: number, heatLoad: number,
  grass: number, water: number, shade: number, neighborPresence: number, socialStress: number,
];

export function chooseSyntheticPolicyAction(observation: CowPolicyObservation, artifact: PolicyArtifact): string {
  if (observation.length !== artifact.input_fields.length) throw new Error('Policy observation schema mismatch');
  let activations = [...observation];
  artifact.layers.forEach((layer, layerIndex) => {
    if (layer.weights_out_in.length !== layer.bias.length) throw new Error('Invalid policy layer');
    const next = layer.weights_out_in.map((weights, outIndex) => {
      if (weights.length !== activations.length) throw new Error('Invalid policy layer width');
      const value = weights.reduce((sum, weight, inIndex) => sum + weight * activations[inIndex], layer.bias[outIndex]);
      return layerIndex === artifact.layers.length - 1 ? value : Math.max(0, value);
    });
    activations = next;
  });
  let best = 0;
  for (let i = 1; i < activations.length; i++) if (activations[i] > activations[best]) best = i;
  return artifact.actions[best];
}
