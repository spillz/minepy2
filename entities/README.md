# Entities

Entities are dynamic objects in the world that are not part of the static block-based terrain. This includes the player, animals, monsters, and any other moving or interactive objects. All entities consist of a tree of sized blocks from a root node with pivot and position offsets. This makes them relatively efficient to render. While blocks cannot be angle in their definiton, animation defines orientation ranges of children relative to parents in different animation states.

## Creating a New Entity

There are two ways to create a new entity: by using the `tetrapod` class as a starting point (recommended for four-legged creatures), or by creating one from scratch.

### Method 1: Using the Tetrapod Class (Recommended)

If you want to create a four-legged creature, the easiest way is to subclass the `Tetrapod` class located in `entities/tetrapod.py`. This class provides a pre-built model and basic animations for a tetrapod, which you can then customize. The `dog.py` entity is a good example of this approach.

Here's how to create your own creature, for example, a `cat.py`:

1.  **Create a new file:** Create a new file in the `entities` directory, for example, `entities/cat.py`.

2.  **Define a model builder function:** In your new file, create a function that calls `build_tetrapod_model` from `tetrapod.py` with your desired customizations. You can change the size, color, and other properties of the body parts.

    ```python
    # entities/cat.py
    from __future__ import annotations
    from typing import Dict, Any
    import numpy as np
    from entities.tetrapod import build_tetrapod_model, Tetrapod

    def build_cat_model() -> Dict[str, Any]:
        return build_tetrapod_model(
            body_size=(0.25, 0.25, 0.5),
            leg_size=(0.08, 0.25, 0.08),
            head_size=(0.2, 0.2, 0.2),
            ear_size=(0.03, 0.08, 0.03),
            snout_size=(0.15, 0.06, 0.15),
            tail_size=(0.03, 0.03, 0.25),
            body_color=(150, 150, 150), # Grey
            leg_color=(150, 150, 150),
            head_color=(150, 150, 150),
            ear_color=(140, 140, 140),
            snout_color=(150, 150, 150),
            tail_color=(150, 150, 150),
        )

    CAT_MODEL = build_cat_model()
    ```

3.  **Create the entity class:** Create a class for your entity that inherits from `Tetrapod`.

    ```python
    # entities/cat.py (continued)
    class Cat(Tetrapod):
        def __init__(self, world, entity_id=5, saved_state=None):
            super().__init__(
                world,
                entity_type="cat",
                model_definition=CAT_MODEL,
                entity_id=entity_id,
                saved_state=saved_state,
            )

        def update(self, dt, context):
            # Add custom cat behavior here
            super().update(dt, context) # Call parent update for physics
    ```

4.  **Add the entity to `main.py`:** Follow the example of the other entities in `main.py` to add your new creature to the world.

### Method 2: From Scratch

Creating an entity from scratch gives you complete control over its appearance and behavior. This is more complex but allows for non-tetrapod creatures (like the snake or snail).

#### Model Definition

An entity's appearance is defined by a dictionary, often created by a `build_<entity_name>_model` function. This dictionary has two main keys: `parts` and `animations`.

*   **`parts`**: A dictionary where each key is the name of a body part. Each part is itself a dictionary with the following keys:
    *   `parent`: The name of the parent part, or `None` if it's the root part. The root part is the base of the entity's hierarchy.
    *   `pivot`: The `[x, y, z]` point on the *parent* part where this part is attached and rotates.
    *   `position`: The `[x, y, z]` offset from the `pivot` to the center of this part's mesh.
    *   `size`: The `[width, height, depth]` of the part's box shape.
    *   `material`: A dictionary containing the `color` of the part as an `(R, G, B)` tuple.

*   **`animations`**: A dictionary of animations for the entity, like 'idle' or 'walk'. Each animation has a `length` (in seconds), a `loop` boolean, and a list of `keyframes`. Each keyframe specifies rotations for different parts at a given `time`.

**Coordinate System and Dimensions**

*   The origin `[0, 0, 0]` of a part is at its center.
*   `pivot` and `position` work together to place a part relative to its parent. Imagine a joint (the `pivot`) on the parent's body. The child part is then shifted by the `position` vector from that joint.
*   Dimensions (`size`) are `[width, height, depth]` corresponding to the x, y, and z axes.

#### Entity Class

The entity class defines the behavior of the entity. It should inherit from `BaseEntity` (`from entity import BaseEntity`).

Here are the essential methods to implement:

*   **`__init__(self, world, ...)`**: The constructor. It should call `super().__init__`, set the `model_definition`, and initialize any entity-specific properties.
*   **`update(self, dt, context)`**: This method is called every frame and contains the entity's logic (e.g., movement, AI).
*   **`serialize_state(self)`**: Returns a dictionary of the entity's state (like position and rotation) that should be saved to disk.
*   **`to_network_dict(self)`**: Returns a dictionary of the entity's state for networking. This is used to send entity information to clients.

By following these guidelines, you can create a wide variety of entities to populate the world.
