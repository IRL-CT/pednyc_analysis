# Final interpretation

The target is the observable pedestrian hand-gesture Yes/No annotation. The models do not understand or infer pedestrian intent.

1. **Does skeleton data add measurable value?** Not in the learned models. The skeleton LSTM achieved balanced accuracy 0.485 (95% participant-bootstrap CI 0.421–0.569); handcrafted skeleton logistic regression achieved 0.536. Neither establishes improvement over chance/majority performance. The fixed hand-motion rule reached 0.610, but it is a newly introduced transparent proxy, not an existing study detector, and its macro F1 was only 0.450.
2. **Is vehicle/context data necessary?** Not established. Context-only logistic regression reached 0.576, while fusion fell to 0.463. The context-only interval includes 0.5 and fusion did not improve skeleton performance, so these data do not demonstrate necessary or additive context information for this label.
3. **Do results generalize across participants?** Not established. Evaluation was correctly participant-grouped, but only 25 positive and 113 negative trials were usable. Confidence intervals overlap chance for all learned models.
4. **Is the model stable enough for continued confirmatory research?** No. Fold balanced-accuracy ranges were 0.413–0.601 for skeleton LSTM, 0.363–0.841 for context-only, and 0.348–0.554 for fusion.
5. **Overall status:** The experiment should remain exploratory. A larger positive class, an independently available source workbook, and an external participant cohort are needed before model selection or scientific claims.
