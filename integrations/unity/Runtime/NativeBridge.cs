// Copyright 2025 Tigantic Labs. All Rights Reserved.

using UnityEngine;

namespace Tigantic.Ontic
{
    /// <summary>
    /// Interface for native Ontic bridge implementations.
    /// </summary>
    public interface INativeBridge
    {
        /// <summary>
        /// Initialize a new field.
        /// </summary>
        /// <returns>Field handle, or -1 on failure.</returns>
        int Initialize(Vector3Int gridSize, FieldType fieldType, int bitsPerDim);

        /// <summary>
        /// Shutdown and release field resources.
        /// </summary>
        void Shutdown(int handle);

        /// <summary>
        /// Sample field at normalized coordinates.
        /// </summary>
        Vector4 Sample(int handle, Vector3 fieldCoords);

        /// <summary>
        /// Extract 2D slice from field.
        /// </summary>
        /// <returns>Flattened RGBA float array, or null on failure.</returns>
        float[] Slice(int handle, SliceSpec spec);

        /// <summary>
        /// Step simulation forward.
        /// </summary>
        void Step(int handle, float deltaTime);

        /// <summary>
        /// Get field statistics.
        /// </summary>
        FieldStats GetStats(int handle);

        /// <summary>
        /// Apply impulse to field.
        /// </summary>
        void ApplyImpulse(int handle, Vector3 position, Vector3 direction, float strength, float radius);

        /// <summary>
        /// Apply uniform force.
        /// </summary>
        void ApplyForce(int handle, Vector3 force);

        /// <summary>
        /// Set obstacle mask.
        /// </summary>
        void SetObstacle(int handle, byte[] voxelMask, Vector3Int size);

        /// <summary>
        /// Clear obstacles.
        /// </summary>
        void ClearObstacles(int handle);

        /// <summary>
        /// Save field to bundle file.
        /// </summary>
        bool Save(int handle, string path);

        /// <summary>
        /// Load field from bundle file.
        /// </summary>
        /// <returns>Field handle, or -1 on failure.</returns>
        int Load(string path);
    }

    /// <summary>
    /// ZMQ-based native bridge for Python backend communication.
    /// </summary>
    public class ZmqNativeBridge : INativeBridge
    {
        private bool isConnected;
        private string endpoint;

        // Note: In a real implementation, this would use NetMQ or ZeroMQ binding
        // For now, this is a placeholder that shows the expected interface

        public bool Connect(string endpoint)
        {
            this.endpoint = endpoint;
            
            // TODO: Implement actual ZMQ connection
            // try
            // {
            //     socket = new RequestSocket();
            //     socket.Connect(endpoint);
            //     isConnected = true;
            // }
            // catch { isConnected = false; }
            
            return isConnected;
        }

        public void Disconnect()
        {
            // TODO: Close ZMQ socket
            isConnected = false;
        }

        public int Initialize(Vector3Int gridSize, FieldType fieldType, int bitsPerDim)
        {
            if (!isConnected) return -1;

            // TODO: Send INIT message via ZMQ
            // var config = new { size_x = gridSize.x, size_y = gridSize.y, size_z = gridSize.z, field_type = fieldType.ToString().ToLower() };
            // SendMessage(MessageType.INIT, 0, JsonConvert.SerializeObject(config));
            // var response = ReceiveResponse();
            // return response.handle;

            return -1;
        }

        public void Shutdown(int handle)
        {
            if (!isConnected) return;
            // TODO: Send SHUTDOWN message
        }

        public Vector4 Sample(int handle, Vector3 fieldCoords)
        {
            if (!isConnected) return Vector4.zero;
            // TODO: Send SAMPLE message and parse response
            return Vector4.zero;
        }

        public float[] Slice(int handle, SliceSpec spec)
        {
            if (!isConnected) return null;
            // TODO: Send SLICE message and parse response
            return null;
        }

        public void Step(int handle, float deltaTime)
        {
            if (!isConnected) return;
            // TODO: Send STEP message
        }

        public FieldStats GetStats(int handle)
        {
            if (!isConnected) return default;
            // TODO: Send STATS message and parse response
            return default;
        }

        public void ApplyImpulse(int handle, Vector3 position, Vector3 direction, float strength, float radius)
        {
            if (!isConnected) return;
            // TODO: Send IMPULSE message
        }

        public void ApplyForce(int handle, Vector3 force)
        {
            if (!isConnected) return;
            // TODO: Send FORCE message
        }

        public void SetObstacle(int handle, byte[] voxelMask, Vector3Int size)
        {
            if (!isConnected) return;
            // TODO: Send OBSTACLE message
        }

        public void ClearObstacles(int handle)
        {
            if (!isConnected) return;
            // TODO: Send CLEAR_OBSTACLES message
        }

        public bool Save(int handle, string path)
        {
            if (!isConnected) return false;
            // TODO: Send SAVE message
            return false;
        }

        public int Load(string path)
        {
            if (!isConnected) return -1;
            // TODO: Send LOAD message and parse response
            return -1;
        }
    }
}
