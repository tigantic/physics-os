/**
 * Ontic Facial Plastics — Three.js Loader
 *
 * Loads Three.js and OrbitControls from the self-hosted importmap,
 * creates a mutable proxy (ES module namespaces are sealed),
 * and exposes window.THREE for non-module scripts.
 */

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

// ES module namespace objects are frozen — cannot add properties.
// Create a mutable proxy that downstream code attaches to.
const proxy = Object.assign({}, THREE);
proxy.OrbitControls = OrbitControls;
window.THREE = proxy;
window.dispatchEvent(new Event("three-ready"));
