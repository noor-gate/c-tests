#include "ann.h"

/* Creates and returns a new ann. */
ann_t *ann_create(int num_layers, int *layer_outputs)
{
  ann_t *new = malloc(sizeof(ann_t));
  if (!new) {
    return NULL;
  } 
  layer_t *curr = layer_create();
  layer_t *previous = NULL;
  for (int i = 0; i < num_layers; i++) {
    bool init = layer_init(curr, layer_outputs[i], previous);
    if (init) {
      return NULL;
    }
    if (i == 0) {
      new->input_layer = curr;
    }
    layer_t *next = layer_create();
    curr->prev = previous;
    if (i == num_layers - 1) {
      curr->next = NULL;
    } else { 
      curr->next = next;
    }  
    previous = curr;
    curr = next;
  }
  new->output_layer = previous;
  return new;
}

/* Frees the space allocated to ann. */
void ann_free(ann_t *ann)
{
  layer_t *curr = ann->input_layer;
  while (curr != NULL) {
    layer_free(curr);
    curr = curr->next;
  }  
  free(ann);
}

/* Forward run of given ann with inputs. */
void ann_predict(ann_t const *ann, double const *inputs)
{
  layer_t *curr = ann->input_layer;
  for (int i = 0; i < curr->num_outputs; i++) {
    curr->outputs[i] = inputs[i];
  } 
  curr = curr->next;
  while (curr != NULL) {
    layer_compute_outputs(curr);
    curr = curr->next;
  }  
}

/* Trains the ann with single backprop update. */
void ann_train(ann_t const *ann, double const *inputs, double const *targets, double l_rate)
{
  /* Sanity checks. */
  assert(ann != NULL);
  assert(inputs != NULL);
  assert(targets != NULL);
  assert(l_rate > 0);

  /* Run forward pass. */
  ann_predict(ann, inputs);

  for (int j = 0; j < (ann->output_layer)->num_outputs; j++) {
    (ann->output_layer)->deltas[j] = sigmoidprime((ann->output_layer)->outputs[j]) * (targets[j] - ((ann->output_layer)->outputs[j]));
  }  

  layer_t *curr = (ann->input_layer)->next;
  while (curr != NULL) {
    if (curr->next != NULL) {
      layer_compute_deltas(curr);
    }
    layer_update(curr, l_rate);
    curr = curr->next;
  }  

}
