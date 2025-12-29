// Copyright 2025 Tigantic Labs. All Rights Reserved.

using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;

namespace Tigantic.HyperTensor
{
    /// <summary>
    /// Event fired when field is updated.
    /// </summary>
    [Serializable]
    public class FieldUpdatedEvent : UnityEvent<FieldStats> { }

    /// <summary>
    /// HyperTensor Field MonoBehaviour.
    /// 
    /// Manages a QTT-compressed field within Unity.
    /// Connects to HyperTensor Python backend for computation.
    /// </summary>
    [AddComponentMenu("HyperTensor/Field")]
    [DisallowMultipleComponent]
    public class HyperTensorField : MonoBehaviour
    {
        // =========================================================================
        // CONFIGURATION
        // =========================================================================

        [Header("Field Configuration")]
        [SerializeField]
        [Tooltip("Type of field (Scalar/Vector/Tensor)")]
        private FieldType fieldType = FieldType.Vector;

        [SerializeField]
        [Tooltip("Grid dimensions")]
        private Vector3Int gridSize = new Vector3Int(64, 64, 64);

        [SerializeField]
        [Tooltip("World-space bounds of the field")]
        private Bounds worldBounds = new Bounds(Vector3.zero, Vector3.one * 10f);

        [SerializeField]
        [Tooltip("Boundary condition")]
        private BoundaryCondition boundaryCondition = BoundaryCondition.Periodic;

        [Header("Physics")]
        [SerializeField]
        private PhysicsConfig physicsConfig = PhysicsConfig.Default;

        [Header("Budget")]
        [SerializeField]
        private BudgetConfig budgetConfig = BudgetConfig.Default;

        [Header("Simulation")]
        [SerializeField]
        [Tooltip("Automatically step simulation each frame")]
        private bool autoStep = true;

        [SerializeField]
        [Tooltip("Time scale for simulation")]
        private float timeScale = 1.0f;

        [Header("Events")]
        public FieldUpdatedEvent OnFieldUpdated = new FieldUpdatedEvent();

        // =========================================================================
        // PROPERTIES
        // =========================================================================

        /// <summary>Field type.</summary>
        public FieldType FieldType => fieldType;

        /// <summary>Grid dimensions.</summary>
        public Vector3Int GridSize => gridSize;

        /// <summary>World-space bounds.</summary>
        public Bounds WorldBounds
        {
            get => worldBounds;
            set => worldBounds = value;
        }

        /// <summary>Physics configuration.</summary>
        public PhysicsConfig PhysicsConfig
        {
            get => physicsConfig;
            set => physicsConfig = value;
        }

        /// <summary>Budget configuration.</summary>
        public BudgetConfig BudgetConfig
        {
            get => budgetConfig;
            set => budgetConfig = value;
        }

        /// <summary>Check if field is initialized.</summary>
        public bool IsInitialized => isInitialized;

        /// <summary>Current field statistics.</summary>
        public FieldStats Stats => cachedStats;

        // =========================================================================
        // PRIVATE FIELDS
        // =========================================================================

        private bool isInitialized;
        private FieldStats cachedStats;
        private INativeBridge nativeBridge;
        private int fieldHandle = -1;

        // Cache for sampling
        private readonly List<Vector3> samplePointsCache = new List<Vector3>();
        private readonly List<Vector4> sampleResultsCache = new List<Vector4>();

        // =========================================================================
        // LIFECYCLE
        // =========================================================================

        private void Awake()
        {
            // Try to initialize native bridge
            nativeBridge = CreateNativeBridge();
        }

        private void Start()
        {
            if (gridSize.x > 0 && gridSize.y > 0 && gridSize.z > 0)
            {
                Initialize(gridSize.x, gridSize.y, gridSize.z, fieldType);
            }
        }

        private void Update()
        {
            if (isInitialized && autoStep)
            {
                Step(Time.deltaTime * timeScale);
            }
        }

        private void OnDestroy()
        {
            if (isInitialized && nativeBridge != null)
            {
                nativeBridge.Shutdown(fieldHandle);
            }
        }

        private void OnDrawGizmosSelected()
        {
            // Draw field bounds
            Gizmos.color = new Color(0, 1, 1, 0.3f);
            Gizmos.matrix = transform.localToWorldMatrix;
            Gizmos.DrawWireCube(worldBounds.center, worldBounds.size);
        }

        // =========================================================================
        // INITIALIZATION
        // =========================================================================

        /// <summary>
        /// Initialize the field with given dimensions.
        /// </summary>
        public bool Initialize(int sizeX, int sizeY, int sizeZ, FieldType type = FieldType.Vector)
        {
            if (isInitialized)
            {
                Debug.LogWarning("HyperTensorField already initialized");
                return false;
            }

            gridSize = new Vector3Int(sizeX, sizeY, sizeZ);
            fieldType = type;

            // Calculate bits per dimension
            int bits = Mathf.CeilToInt(Mathf.Log(Mathf.Max(sizeX, sizeY, sizeZ), 2));

            // Initialize via native bridge if available
            if (nativeBridge != null)
            {
                fieldHandle = nativeBridge.Initialize(gridSize, type, bits);
                if (fieldHandle < 0)
                {
                    Debug.LogError("Failed to initialize field via native bridge");
                    return false;
                }
            }

            // Initialize cached stats
            cachedStats = new FieldStats
            {
                maxRank = budgetConfig.maxRank,
                numCores = bits * 3, // 3D field
                compressionRatio = 100f,
                stepCount = 0
            };

            isInitialized = true;
            Debug.Log($"HyperTensorField initialized: {sizeX}x{sizeY}x{sizeZ}, Type={type}");

            return true;
        }

        /// <summary>
        /// Initialize from a saved bundle file.
        /// </summary>
        public bool InitializeFromBundle(string bundlePath)
        {
            if (nativeBridge == null)
            {
                Debug.LogError("Native bridge not available");
                return false;
            }

            fieldHandle = nativeBridge.Load(bundlePath);
            if (fieldHandle >= 0)
            {
                isInitialized = true;
                UpdateCachedStats();
                return true;
            }

            return false;
        }

        // =========================================================================
        // FIELD ORACLE API
        // =========================================================================

        /// <summary>
        /// Sample field at world positions.
        /// </summary>
        public SampleResult Sample(Vector3[] worldPositions)
        {
            var result = new SampleResult
            {
                values = new Vector4[worldPositions.Length],
                rankUsed = budgetConfig.maxRank,
                errorEstimate = 0f
            };

            if (!isInitialized) return result;

            for (int i = 0; i < worldPositions.Length; i++)
            {
                result.values[i] = SampleSingle(worldPositions[i]);
            }

            return result;
        }

        /// <summary>
        /// Sample field at a single world position.
        /// </summary>
        public Vector4 Sample(Vector3 worldPosition)
        {
            return SampleSingle(worldPosition);
        }

        /// <summary>
        /// Sample field at a single world position.
        /// </summary>
        public Vector4 SampleSingle(Vector3 worldPosition)
        {
            if (!isInitialized)
            {
                return Vector4.zero;
            }

            Vector3 fieldCoords = WorldToFieldCoords(worldPosition);

            // Clamp to field bounds
            fieldCoords.x = Mathf.Clamp01(fieldCoords.x);
            fieldCoords.y = Mathf.Clamp01(fieldCoords.y);
            fieldCoords.z = Mathf.Clamp01(fieldCoords.z);

            // Use native bridge if available
            if (nativeBridge != null)
            {
                return nativeBridge.Sample(fieldHandle, fieldCoords);
            }

            // Placeholder sampling based on position
            float dist = (fieldCoords - Vector3.one * 0.5f).magnitude;
            float value = Mathf.Exp(-dist * 4f);

            return fieldType switch
            {
                FieldType.Scalar => new Vector4(value, 0, 0, 0),
                FieldType.Vector => new Vector4(
                    value * (0.5f - fieldCoords.x),
                    value * (0.5f - fieldCoords.y),
                    value * (0.5f - fieldCoords.z),
                    0
                ),
                FieldType.Tensor => new Vector4(value, value, value, value),
                _ => Vector4.zero
            };
        }

        /// <summary>
        /// Extract 2D slice from field.
        /// </summary>
        public Texture2D Slice(SliceSpec spec)
        {
            if (!isInitialized)
            {
                return null;
            }

            var texture = new Texture2D(spec.resolution.x, spec.resolution.y, TextureFormat.RGBAFloat, false);

            // Use native bridge if available
            if (nativeBridge != null)
            {
                float[] data = nativeBridge.Slice(fieldHandle, spec);
                if (data != null)
                {
                    // Convert to texture
                    Color[] colors = new Color[spec.resolution.x * spec.resolution.y];
                    for (int i = 0; i < colors.Length; i++)
                    {
                        int idx = i * 4;
                        colors[i] = new Color(data[idx], data[idx + 1], data[idx + 2], data[idx + 3]);
                    }
                    texture.SetPixels(colors);
                    texture.Apply();
                }
            }

            return texture;
        }

        /// <summary>
        /// Step the simulation forward.
        /// </summary>
        public void Step(float deltaTime)
        {
            if (!isInitialized) return;

            // Use native bridge if available
            if (nativeBridge != null)
            {
                nativeBridge.Step(fieldHandle, deltaTime);
                UpdateCachedStats();
            }
            else
            {
                // Increment step count for placeholder
                cachedStats.stepCount++;
            }

            // Fire event
            OnFieldUpdated?.Invoke(cachedStats);
        }

        // =========================================================================
        // SIMULATION CONTROLS
        // =========================================================================

        /// <summary>
        /// Apply an impulse to the field.
        /// </summary>
        public void ApplyImpulse(Vector3 worldPosition, Vector3 direction, float strength, float radius)
        {
            if (!isInitialized) return;

            Vector3 fieldCoords = WorldToFieldCoords(worldPosition);
            float normalizedRadius = radius / worldBounds.size.magnitude;

            if (nativeBridge != null)
            {
                nativeBridge.ApplyImpulse(fieldHandle, fieldCoords, direction.normalized, strength, normalizedRadius);
            }

            Debug.Log($"ApplyImpulse at {fieldCoords}, Strength={strength}, Radius={normalizedRadius}");
        }

        /// <summary>
        /// Apply a force field for one frame.
        /// </summary>
        public void ApplyForce(Vector3 forceField)
        {
            if (!isInitialized) return;

            physicsConfig.externalForce = forceField;

            if (nativeBridge != null)
            {
                nativeBridge.ApplyForce(fieldHandle, forceField);
            }
        }

        /// <summary>
        /// Set obstacle from mesh collider.
        /// </summary>
        public void SetObstacle(MeshCollider meshCollider)
        {
            if (!isInitialized || meshCollider == null) return;

            // TODO: Voxelize mesh and send to backend
            Debug.Log($"SetObstacle: {meshCollider.sharedMesh.name}");
        }

        /// <summary>
        /// Clear all obstacles.
        /// </summary>
        public void ClearObstacles()
        {
            if (!isInitialized) return;

            if (nativeBridge != null)
            {
                nativeBridge.ClearObstacles(fieldHandle);
            }
        }

        // =========================================================================
        // SERIALIZATION
        // =========================================================================

        /// <summary>
        /// Save field to bundle file.
        /// </summary>
        public bool SaveToBundle(string bundlePath)
        {
            if (!isInitialized || nativeBridge == null)
            {
                return false;
            }

            return nativeBridge.Save(fieldHandle, bundlePath);
        }

        /// <summary>
        /// Get field state hash for provenance.
        /// </summary>
        public string GetStateHash()
        {
            return cachedStats.stateHash;
        }

        // =========================================================================
        // COORDINATE CONVERSION
        // =========================================================================

        /// <summary>
        /// Convert world position to normalized field coordinates.
        /// </summary>
        public Vector3 WorldToFieldCoords(Vector3 worldPos)
        {
            // Transform to local space first
            Vector3 localPos = transform.InverseTransformPoint(worldPos);
            
            Vector3 min = worldBounds.min;
            Vector3 size = worldBounds.size;

            return new Vector3(
                (localPos.x - min.x) / size.x,
                (localPos.y - min.y) / size.y,
                (localPos.z - min.z) / size.z
            );
        }

        /// <summary>
        /// Convert normalized field coordinates to world position.
        /// </summary>
        public Vector3 FieldCoordsToWorld(Vector3 fieldCoords)
        {
            Vector3 min = worldBounds.min;
            Vector3 size = worldBounds.size;

            Vector3 localPos = new Vector3(
                min.x + fieldCoords.x * size.x,
                min.y + fieldCoords.y * size.y,
                min.z + fieldCoords.z * size.z
            );

            return transform.TransformPoint(localPos);
        }

        // =========================================================================
        // INTERNAL
        // =========================================================================

        private void UpdateCachedStats()
        {
            if (nativeBridge != null)
            {
                cachedStats = nativeBridge.GetStats(fieldHandle);
            }
        }

        private INativeBridge CreateNativeBridge()
        {
            // Try ZMQ bridge first
            try
            {
                var bridge = new ZmqNativeBridge();
                if (bridge.Connect("tcp://localhost:5555"))
                {
                    Debug.Log("Connected to HyperTensor Python backend via ZMQ");
                    return bridge;
                }
            }
            catch (Exception e)
            {
                Debug.LogWarning($"ZMQ bridge not available: {e.Message}");
            }

            // Fall back to placeholder
            Debug.Log("Using placeholder field implementation");
            return null;
        }
    }
}
