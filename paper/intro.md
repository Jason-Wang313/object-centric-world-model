# Introduction

Object-centric world models promise predictions organized around persistent entities. That organization is useful only if the selected future binds the right object. Best-of-N inference makes the binding problem sharper: selection focuses attention on the upper tail of a model's score distribution, exactly where plausible-looking artifacts can be overrepresented.

This paper asks what happens when objects lie. A model may assign high score to a future that appears coherent in slot space but swaps the target with a visually similar distractor, merges two objects, loses identity through occlusion, or assumes the wrong hidden mass. The resulting future can be attractive to the selector and poor under real utility.

Our contribution is a compact controlled study: an exact finite selection law, object-centric synthetic environments, a learned CPU artifact with identity alignment, repair selectors, paired and stress evidence, and an audit that keeps unsupported claims unsupported.
