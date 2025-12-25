// Copyright 2025 Tigantic Labs. All Rights Reserved.

using UnityEngine;
using UnityEditor;

namespace Tigantic.HyperTensor.Editor
{
    /// <summary>
    /// Custom inspector for HyperTensorField.
    /// </summary>
    [CustomEditor(typeof(HyperTensorField))]
    public class HyperTensorFieldEditor : UnityEditor.Editor
    {
        private HyperTensorField field;
        private bool showStats = true;
        private bool showPhysics = true;
        private bool showBudget = true;

        private void OnEnable()
        {
            field = (HyperTensorField)target;
        }

        public override void OnInspectorGUI()
        {
            serializedObject.Update();

            // Header
            EditorGUILayout.Space();
            EditorGUILayout.LabelField("HyperTensor Field", EditorStyles.boldLabel);
            EditorGUILayout.Space();

            // Status
            using (new EditorGUILayout.HorizontalScope())
            {
                EditorGUILayout.LabelField("Status:", GUILayout.Width(60));
                if (field.IsInitialized)
                {
                    EditorGUILayout.LabelField("Initialized", EditorStyles.boldLabel);
                }
                else
                {
                    EditorGUILayout.LabelField("Not Initialized", EditorStyles.miniLabel);
                }
            }

            EditorGUILayout.Space();

            // Configuration section
            EditorGUILayout.LabelField("Configuration", EditorStyles.boldLabel);
            EditorGUI.indentLevel++;

            EditorGUILayout.PropertyField(serializedObject.FindProperty("fieldType"));
            EditorGUILayout.PropertyField(serializedObject.FindProperty("gridSize"));
            EditorGUILayout.PropertyField(serializedObject.FindProperty("worldBounds"));
            EditorGUILayout.PropertyField(serializedObject.FindProperty("boundaryCondition"));

            EditorGUI.indentLevel--;
            EditorGUILayout.Space();

            // Physics section
            showPhysics = EditorGUILayout.Foldout(showPhysics, "Physics", true);
            if (showPhysics)
            {
                EditorGUI.indentLevel++;
                EditorGUILayout.PropertyField(serializedObject.FindProperty("physicsConfig"));
                EditorGUI.indentLevel--;
            }

            EditorGUILayout.Space();

            // Budget section
            showBudget = EditorGUILayout.Foldout(showBudget, "Budget", true);
            if (showBudget)
            {
                EditorGUI.indentLevel++;
                EditorGUILayout.PropertyField(serializedObject.FindProperty("budgetConfig"));
                EditorGUI.indentLevel--;
            }

            EditorGUILayout.Space();

            // Simulation section
            EditorGUILayout.LabelField("Simulation", EditorStyles.boldLabel);
            EditorGUI.indentLevel++;
            EditorGUILayout.PropertyField(serializedObject.FindProperty("autoStep"));
            EditorGUILayout.PropertyField(serializedObject.FindProperty("timeScale"));
            EditorGUI.indentLevel--;

            EditorGUILayout.Space();

            // Stats section (runtime only)
            if (Application.isPlaying && field.IsInitialized)
            {
                showStats = EditorGUILayout.Foldout(showStats, "Runtime Stats", true);
                if (showStats)
                {
                    EditorGUI.indentLevel++;
                    DrawStats();
                    EditorGUI.indentLevel--;
                }
            }

            EditorGUILayout.Space();

            // Events
            EditorGUILayout.PropertyField(serializedObject.FindProperty("OnFieldUpdated"));

            serializedObject.ApplyModifiedProperties();

            // Repaint for live updates
            if (Application.isPlaying)
            {
                Repaint();
            }
        }

        private void DrawStats()
        {
            var stats = field.Stats;

            using (new EditorGUILayout.VerticalScope(EditorStyles.helpBox))
            {
                EditorGUILayout.LabelField($"Max Rank: {stats.maxRank}");
                EditorGUILayout.LabelField($"Avg Rank: {stats.avgRank:F2}");
                EditorGUILayout.LabelField($"Cores: {stats.numCores}");
                EditorGUILayout.LabelField($"Truncation Error: {stats.truncationError:E2}");
                EditorGUILayout.LabelField($"Energy: {stats.energy:F4}");
                EditorGUILayout.LabelField($"Compression: {stats.compressionRatio:F1}x");
                EditorGUILayout.LabelField($"Memory: {stats.memoryBytes / 1024f:F1} KB");
                EditorGUILayout.LabelField($"Step Count: {stats.stepCount}");

                if (!string.IsNullOrEmpty(stats.stateHash))
                {
                    EditorGUILayout.LabelField($"State Hash: {stats.stateHash}");
                }
            }
        }

        private void OnSceneGUI()
        {
            // Draw bounds handle
            EditorGUI.BeginChangeCheck();

            var boundsProperty = serializedObject.FindProperty("worldBounds");
            var bounds = boundsProperty.boundsValue;

            var center = Handles.PositionHandle(
                field.transform.TransformPoint(bounds.center),
                field.transform.rotation
            );

            if (EditorGUI.EndChangeCheck())
            {
                Undo.RecordObject(field, "Move Field Bounds");
                bounds.center = field.transform.InverseTransformPoint(center);
                boundsProperty.boundsValue = bounds;
                serializedObject.ApplyModifiedProperties();
            }

            // Draw bounds wireframe
            Handles.color = new Color(0, 1, 1, 0.5f);
            var worldBounds = new Bounds(
                field.transform.TransformPoint(bounds.center),
                Vector3.Scale(bounds.size, field.transform.lossyScale)
            );
            Handles.DrawWireCube(worldBounds.center, worldBounds.size);
        }
    }
}
