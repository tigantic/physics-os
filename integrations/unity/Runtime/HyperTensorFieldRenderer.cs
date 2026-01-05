// Copyright 2025 Tigantic Labs. All Rights Reserved.

using UnityEngine;

namespace Tigantic.HyperTensor
{
    /// <summary>
    /// Field renderer for visualizing HyperTensor fields.
    /// Supports slices, isosurfaces, and volume rendering.
    /// </summary>
    [RequireComponent(typeof(HyperTensorField))]
    [AddComponentMenu("HyperTensor/Field Renderer")]
    public class HyperTensorFieldRenderer : MonoBehaviour
    {
        // =========================================================================
        // CONFIGURATION
        // =========================================================================

        [Header("Visualization Mode")]
        [SerializeField]
        private VisualizationMode mode = VisualizationMode.Slice;

        [Header("Slice Settings")]
        [SerializeField]
        [Range(0, 2)]
        private int sliceAxis = 2;

        [SerializeField]
        [Range(0f, 1f)]
        private float slicePosition = 0.5f;

        [SerializeField]
        private Vector2Int sliceResolution = new Vector2Int(256, 256);

        [Header("Volume Settings")]
        [SerializeField]
        private int volumeResolution = 64;

        [SerializeField]
        [Range(0f, 1f)]
        private float volumeDensity = 0.5f;

        [Header("Colormap")]
        [SerializeField]
        private Gradient colormap;

        [SerializeField]
        [Range(-1f, 1f)]
        private float valueMin = -1f;

        [SerializeField]
        [Range(-1f, 1f)]
        private float valueMax = 1f;

        [Header("Materials")]
        [SerializeField]
        private Material sliceMaterial;

        [SerializeField]
        private Material volumeMaterial;

        // =========================================================================
        // PROPERTIES
        // =========================================================================

        public VisualizationMode Mode
        {
            get => mode;
            set
            {
                mode = value;
                UpdateVisualization();
            }
        }

        public float SlicePosition
        {
            get => slicePosition;
            set
            {
                slicePosition = Mathf.Clamp01(value);
                UpdateSlice();
            }
        }

        // =========================================================================
        // PRIVATE FIELDS
        // =========================================================================

        private HyperTensorField field;
        private MeshRenderer sliceRenderer;
        private MeshFilter sliceMeshFilter;
        private Texture2D sliceTexture;
        private Material sliceMaterialInstance;

        // =========================================================================
        // LIFECYCLE
        // =========================================================================

        private void Awake()
        {
            field = GetComponent<HyperTensorField>();

            // Setup default colormap if not set
            if (colormap == null)
            {
                colormap = new Gradient();
                colormap.SetKeys(
                    new GradientColorKey[]
                    {
                        new GradientColorKey(Color.blue, 0f),
                        new GradientColorKey(Color.white, 0.5f),
                        new GradientColorKey(Color.red, 1f)
                    },
                    new GradientAlphaKey[]
                    {
                        new GradientAlphaKey(1f, 0f),
                        new GradientAlphaKey(1f, 1f)
                    }
                );
            }
        }

        private void Start()
        {
            SetupVisualization();
        }

        private void OnEnable()
        {
            if (field != null)
            {
                field.OnFieldUpdated.AddListener(OnFieldUpdated);
            }
        }

        private void OnDisable()
        {
            if (field != null)
            {
                field.OnFieldUpdated.RemoveListener(OnFieldUpdated);
            }
        }

        private void OnDestroy()
        {
            if (sliceTexture != null)
            {
                Destroy(sliceTexture);
            }
            if (sliceMaterialInstance != null)
            {
                Destroy(sliceMaterialInstance);
            }
        }

        // =========================================================================
        // SETUP
        // =========================================================================

        private void SetupVisualization()
        {
            switch (mode)
            {
                case VisualizationMode.Slice:
                    SetupSliceVisualization();
                    break;
                case VisualizationMode.Volume:
                    SetupVolumeVisualization();
                    break;
                case VisualizationMode.Isosurface:
                    SetupIsosurfaceVisualization();
                    break;
            }
        }

        private void SetupSliceVisualization()
        {
            // Create child object for slice quad
            var sliceObj = new GameObject("SliceQuad");
            sliceObj.transform.SetParent(transform);
            sliceObj.transform.localPosition = Vector3.zero;
            sliceObj.transform.localRotation = Quaternion.identity;

            // Add mesh components
            sliceMeshFilter = sliceObj.AddComponent<MeshFilter>();
            sliceRenderer = sliceObj.AddComponent<MeshRenderer>();

            // Create quad mesh
            sliceMeshFilter.mesh = CreateQuadMesh();

            // Setup material
            if (sliceMaterial != null)
            {
                sliceMaterialInstance = new Material(sliceMaterial);
                sliceRenderer.material = sliceMaterialInstance;
            }
            else
            {
                // Find shader with fallback for builds
                var shader = Shader.Find("Unlit/Texture");
                if (shader == null)
                {
                    Debug.LogError("HyperTensorFieldRenderer: 'Unlit/Texture' shader not found. Ensure it's included in Always Included Shaders.");
                    shader = Shader.Find("Hidden/InternalErrorShader");
                }
                sliceMaterialInstance = new Material(shader);
                sliceRenderer.material = sliceMaterialInstance;
            }

            // Create texture
            sliceTexture = new Texture2D(sliceResolution.x, sliceResolution.y, TextureFormat.RGBAFloat, false);
            sliceTexture.filterMode = FilterMode.Bilinear;
            sliceMaterialInstance.mainTexture = sliceTexture;

            UpdateSlice();
        }

        private void SetupVolumeVisualization()
        {
            // TODO: Setup volume rendering with ray marching shader
            Debug.Log("Volume visualization not yet implemented");
        }

        private void SetupIsosurfaceVisualization()
        {
            // TODO: Setup marching cubes isosurface extraction
            Debug.Log("Isosurface visualization not yet implemented");
        }

        // =========================================================================
        // UPDATE
        // =========================================================================

        private void OnFieldUpdated(FieldStats stats)
        {
            UpdateVisualization();
        }

        private void UpdateVisualization()
        {
            switch (mode)
            {
                case VisualizationMode.Slice:
                    UpdateSlice();
                    break;
                case VisualizationMode.Volume:
                    UpdateVolume();
                    break;
                case VisualizationMode.Isosurface:
                    UpdateIsosurface();
                    break;
            }
        }

        private void UpdateSlice()
        {
            if (!field.IsInitialized || sliceTexture == null) return;

            var spec = new SliceSpec(sliceAxis, slicePosition, sliceResolution);
            
            // Sample field for slice
            var colors = new Color[sliceResolution.x * sliceResolution.y];
            var bounds = field.WorldBounds;

            for (int y = 0; y < sliceResolution.y; y++)
            {
                for (int x = 0; x < sliceResolution.x; x++)
                {
                    float u = (float)x / (sliceResolution.x - 1);
                    float v = (float)y / (sliceResolution.y - 1);

                    Vector3 fieldCoords = GetSliceCoords(u, v);
                    Vector3 worldPos = field.FieldCoordsToWorld(fieldCoords);
                    Vector4 value = field.Sample(worldPos);

                    // Map value to color
                    float t = Mathf.InverseLerp(valueMin, valueMax, value.x);
                    colors[y * sliceResolution.x + x] = colormap.Evaluate(t);
                }
            }

            sliceTexture.SetPixels(colors);
            sliceTexture.Apply();

            // Update slice quad position and orientation
            UpdateSliceTransform();
        }

        private void UpdateSliceTransform()
        {
            if (sliceMeshFilter == null) return;

            var bounds = field.WorldBounds;
            var sliceTransform = sliceMeshFilter.transform;

            // Position slice at correct location
            Vector3 center = bounds.center;
            Vector3 size = bounds.size;

            switch (sliceAxis)
            {
                case 0: // X slice (YZ plane)
                    center.x = bounds.min.x + slicePosition * size.x;
                    sliceTransform.localRotation = Quaternion.Euler(0, 90, 0);
                    sliceTransform.localScale = new Vector3(size.z, size.y, 1);
                    break;
                case 1: // Y slice (XZ plane)
                    center.y = bounds.min.y + slicePosition * size.y;
                    sliceTransform.localRotation = Quaternion.Euler(90, 0, 0);
                    sliceTransform.localScale = new Vector3(size.x, size.z, 1);
                    break;
                case 2: // Z slice (XY plane)
                default:
                    center.z = bounds.min.z + slicePosition * size.z;
                    sliceTransform.localRotation = Quaternion.identity;
                    sliceTransform.localScale = new Vector3(size.x, size.y, 1);
                    break;
            }

            sliceTransform.localPosition = center;
        }

        private Vector3 GetSliceCoords(float u, float v)
        {
            return sliceAxis switch
            {
                0 => new Vector3(slicePosition, v, u), // X slice
                1 => new Vector3(u, slicePosition, v), // Y slice
                _ => new Vector3(u, v, slicePosition)  // Z slice
            };
        }

        private void UpdateVolume()
        {
            // TODO: Update volume texture for ray marching
        }

        private void UpdateIsosurface()
        {
            // TODO: Run marching cubes and update mesh
        }

        // =========================================================================
        // HELPERS
        // =========================================================================

        private Mesh CreateQuadMesh()
        {
            var mesh = new Mesh();
            mesh.name = "SliceQuad";

            mesh.vertices = new Vector3[]
            {
                new Vector3(-0.5f, -0.5f, 0),
                new Vector3(0.5f, -0.5f, 0),
                new Vector3(0.5f, 0.5f, 0),
                new Vector3(-0.5f, 0.5f, 0)
            };

            mesh.uv = new Vector2[]
            {
                new Vector2(0, 0),
                new Vector2(1, 0),
                new Vector2(1, 1),
                new Vector2(0, 1)
            };

            mesh.triangles = new int[] { 0, 2, 1, 0, 3, 2 };

            mesh.RecalculateNormals();
            mesh.RecalculateBounds();

            return mesh;
        }

        /// <summary>
        /// Map a value to a color using the configured colormap.
        /// </summary>
        public Color ValueToColor(float value)
        {
            float t = Mathf.InverseLerp(valueMin, valueMax, value);
            return colormap.Evaluate(t);
        }
    }

    /// <summary>
    /// Visualization modes for field rendering.
    /// </summary>
    public enum VisualizationMode
    {
        /// <summary>2D slice through the field.</summary>
        Slice,
        /// <summary>Volume rendering with ray marching.</summary>
        Volume,
        /// <summary>Isosurface extraction.</summary>
        Isosurface
    }
}
