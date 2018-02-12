# Discussion

This document to keep track of questions that need to be brainstormed / choices that need to be made

Any bullet that you may add, please add your initials + date so that we can keep track.

## API choices

* Batch first ordering: while in the math lit it is common to write feature matrices as `(n, p)`, in Flux it is `(p, n)` (and usually in NNs it's like this) which may be more efficient because column-major ordering.
    - question as to whether we could hide this away and do both


## Hierarchy of types

* (TL, 22-01-18) Should MetaModels be a type? I'm tempted to say that an ensemble of regressions is still a regression and same for a classification. `MetaModel` would be a subtype of `SupervisedModel` but I'm not sure it would be useful.

## Flux v KNet

* What's the difference?
