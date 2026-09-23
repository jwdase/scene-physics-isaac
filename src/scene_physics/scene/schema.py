"""The scene file format: prim paths and attribute names, in one place.

Every attribute this package authors lives in the ``sp:`` namespace, so it shows up grouped in the
Isaac Sim property panel ("Raw USD Properties") and never collides with UsdPhysics / PhysX attributes.

Format version history:
  1 - initial Isaac Sim format (this rewrite).
"""

from __future__ import annotations

FORMAT_VERSION = 1

# ----------------------------------------------------------------------------- prim paths
WORLD = "/World"
OBJECTS = "/World/Objects"
PHYSICS_SCENE = "/World/PhysicsScene"
CAMERA = "/World/Camera"
GROUND = "/World/Ground"
MESH_NAME = "mesh"  # child Mesh prim holding an object's geometry
LOOKS_NAME = "Looks"

# ----------------------------------------------------------------------------- scene level (on /World)
SCENE_PREFIX = "sp:scene:"
SCENE_FORMAT = "sp:scene:formatVersion"  # int
SCENE_NAME = "sp:scene:name"
SCENE_EXPERIMENT = "sp:scene:experiment"
SCENE_CONDITION = "sp:scene:condition"
SCENE_CUE = "sp:scene:cue"
SCENE_DESCRIPTION = "sp:scene:description"
SCENE_SOURCE = "sp:scene:source"  # where the scene came from (Blender export, legacy dir, ...)
SCENE_STATUS = "sp:scene:status"  # authoring | finalized
SCENE_CREATED = "sp:scene:created"  # ISO timestamp of the last finalize
SCENE_PHYSICS_CONFIG = "sp:scene:physicsConfig"  # JSON of the PhysicsConfig used to settle
SCENE_SETTLED = "sp:scene:settled"  # bool: truth poses are a rest state of that physics

# ----------------------------------------------------------------------------- camera (on /World/Camera)
CAMERA_PREFIX = "sp:camera:"
CAMERA_NAME = "sp:camera:name"
CAMERA_WIDTH = "sp:camera:width"
CAMERA_HEIGHT = "sp:camera:height"
CAMERA_FOV = "sp:camera:fovDegree"  # vertical field of view
CAMERA_EYE = "sp:camera:eye"
CAMERA_TARGET = "sp:camera:target"
CAMERA_UP = "sp:camera:up"
CAMERA_MAX_DEPTH = "sp:camera:maxDepth"

# ----------------------------------------------------------------------------- object level
ROLE = "sp:role"  # token: static | observed | hidden
ROLE_SOURCE = "sp:roleSource"  # token: measured | authored
TAGS = "sp:tags"  # token[]: target, occluder, decoy, probe, reference, ...
ASSET = "sp:asset"  # library / base asset name, e.g. "jug04"
COLOR = "sp:color"  # color3f: the object's categorical colour (target.png fill, overlays)
SEGMENTATION_ID = "sp:segmentationId"  # int, 0 = background

VIS_FRACTION = "sp:visibility:fraction"  # float: visible / silhouette pixels
VIS_VISIBLE = "sp:visibility:visiblePixels"  # int
VIS_SILHOUETTE = "sp:visibility:silhouettePixels"  # int (amodal, occluders ignored)

PRIOR_POSITIONS = "sp:prior:positions"  # double3[] candidate means
PRIOR_ORIENTATIONS = "sp:prior:orientations"  # quatd[] (USD order w, x, y, z)
PRIOR_WEIGHTS = "sp:prior:weights"  # double[] mixture weights
PRIOR_POS_STD = "sp:prior:posStd"  # double
PRIOR_ROT_STD = "sp:prior:rotStd"  # double
PRIOR_BOUNDS = "sp:prior:bounds"  # double4: x_min, x_max, y_min, y_max
PRIOR_RULE = "sp:prior:rule"  # string: how the prior was generated (or "manual")

SETTLE_DISPLACEMENT = "sp:settle:displacement"  # double, metres moved while settling
SETTLE_ROTATION = "sp:settle:rotation"  # double, degrees rotated while settling
SETTLE_RESIDUAL = "sp:settle:residualDisplacement"  # double, metres moved after settling

ROLES = ("static", "observed", "hidden")
KNOWN_TAGS = ("target", "occluder", "decoy", "probe", "reference", "cue")

# PhysX schema attribute value types (queried from Isaac Sim 6.1), so physics can be authored with
# plain pxr outside Kit. Keys are attribute names, values are Sdf.ValueTypeNames attribute names.
PHYSX_ATTR_TYPES = {
    "physxScene:solverType": "Token",
    "physxScene:broadphaseType": "Token",
    "physxScene:enableGPUDynamics": "Bool",
    "physxScene:enableEnhancedDeterminism": "Bool",
    "physxScene:enableStabilization": "Bool",
    "physxScene:enableCCD": "Bool",
    "physxScene:disableSleeping": "Bool",
    "physxScene:timeStepsPerSecond": "UInt",
    "physxScene:gpuMaxRigidContactCount": "UInt",
    "physxScene:gpuMaxRigidPatchCount": "UInt",
    "physxScene:gpuFoundLostPairsCapacity": "UInt",
    "physxScene:gpuTotalAggregatePairsCapacity": "UInt",
    "physxRigidBody:linearDamping": "Float",
    "physxRigidBody:angularDamping": "Float",
    "physxRigidBody:sleepThreshold": "Float",
    "physxRigidBody:stabilizationThreshold": "Float",
    "physxRigidBody:maxDepenetrationVelocity": "Float",
    "physxRigidBody:solverPositionIterationCount": "Int",
    "physxRigidBody:solverVelocityIterationCount": "Int",
    "physxRigidBody:enableCCD": "Bool",
    "physxCollision:contactOffset": "Float",
    "physxCollision:restOffset": "Float",
    "physxConvexDecompositionCollision:maxConvexHulls": "Int",
    "physxConvexDecompositionCollision:hullVertexLimit": "Int",
    "physxConvexDecompositionCollision:shrinkWrap": "Bool",
    "physxConvexHullCollision:hullVertexLimit": "Int",
    "physxSDFMeshCollision:sdfResolution": "Int",
    "physxMaterial:frictionCombineMode": "Token",
    "physxMaterial:restitutionCombineMode": "Token",
}

PHYSX_API_FOR_PREFIX = {
    "physxScene:": "PhysxSceneAPI",
    "physxRigidBody:": "PhysxRigidBodyAPI",
    "physxCollision:": "PhysxCollisionAPI",
    "physxConvexDecompositionCollision:": "PhysxConvexDecompositionCollisionAPI",
    "physxConvexHullCollision:": "PhysxConvexHullCollisionAPI",
    "physxSDFMeshCollision:": "PhysxSDFMeshCollisionAPI",
    "physxMaterial:": "PhysxMaterialAPI",
}
