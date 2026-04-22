import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const canvas = document.querySelector("#scene");
const modeEl = document.querySelector("#mode");
const selectedEl = document.querySelector("#selected");
const speedEl = document.querySelector("#speed");
const planetNameEl = document.querySelector("#planetName");
const planetStatsEl = document.querySelector("#planetStats");
const speedRange = document.querySelector("#speedRange");
const playButton = document.querySelector("#play");
const followButton = document.querySelector("#follow");
const cinematicButton = document.querySelector("#cinematic");

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x05070f);

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 2200);
camera.position.set(0, 58, 122);

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: "high-performance" });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.maxDistance = 420;
controls.minDistance = 12;

const textureLoader = new THREE.TextureLoader();
const texture = (name) => {
  const map = textureLoader.load(`./assets/${name}`);
  map.colorSpace = THREE.SRGBColorSpace;
  map.anisotropy = renderer.capabilities.getMaxAnisotropy();
  return map;
};

scene.add(new THREE.AmbientLight(0x6e85a6, 0.18));

const sunLight = new THREE.PointLight(0xffe0a3, 4.5, 650, 1.7);
scene.add(sunLight);

const starGeometry = new THREE.BufferGeometry();
const starPositions = [];
for (let i = 0; i < 2600; i += 1) {
  const radius = THREE.MathUtils.randFloat(520, 1150);
  const theta = THREE.MathUtils.randFloat(0, Math.PI * 2);
  const phi = Math.acos(THREE.MathUtils.randFloatSpread(2));
  starPositions.push(
    radius * Math.sin(phi) * Math.cos(theta),
    radius * Math.cos(phi),
    radius * Math.sin(phi) * Math.sin(theta),
  );
}
starGeometry.setAttribute("position", new THREE.Float32BufferAttribute(starPositions, 3));
scene.add(
  new THREE.Points(
    starGeometry,
    new THREE.PointsMaterial({ color: 0xcfe7ff, size: 1.7, sizeAttenuation: true, transparent: true, opacity: 0.86 }),
  ),
);

const planets = [
  { name: "Mercury", radius: 1.3, orbit: 12, speed: 4.15, rotation: 2.5, inclination: 7, file: "mercury.jpg" },
  { name: "Venus", radius: 2.2, orbit: 18, speed: 1.62, rotation: 1.1, inclination: 3.4, file: "venus.jpg" },
  { name: "Earth", radius: 2.35, orbit: 25, speed: 1, rotation: 3.2, inclination: 0, file: "earth.jpg" },
  { name: "Mars", radius: 1.75, orbit: 33, speed: 0.53, rotation: 2.8, inclination: 1.85, file: "mars.jpg" },
  { name: "Jupiter", radius: 6.2, orbit: 50, speed: 0.084, rotation: 5.4, inclination: 1.3, file: "jupiter.jpg" },
  { name: "Saturn", radius: 5.25, orbit: 70, speed: 0.034, rotation: 4.8, inclination: 2.5, file: "saturn.jpg" },
  { name: "Uranus", radius: 3.9, orbit: 90, speed: 0.012, rotation: 3.4, inclination: 0.8, file: "uranus.jpg" },
  { name: "Neptune", radius: 3.8, orbit: 108, speed: 0.006, rotation: 3.5, inclination: 1.8, file: "neptune.jpg" },
];

const sphere = new THREE.SphereGeometry(1, 72, 36);
const orbitMaterial = new THREE.LineBasicMaterial({ color: 0x79aef0, transparent: true, opacity: 0.25 });
const bodyGroup = new THREE.Group();
scene.add(bodyGroup);

const sun = new THREE.Mesh(
  new THREE.SphereGeometry(7.5, 96, 48),
  new THREE.MeshBasicMaterial({ map: texture("sun.jpg"), color: 0xffd27a }),
);
bodyGroup.add(sun);

const sunGlow = new THREE.Mesh(
  new THREE.SphereGeometry(8.7, 96, 48),
  new THREE.MeshBasicMaterial({ color: 0xffb84d, transparent: true, opacity: 0.16, blending: THREE.AdditiveBlending }),
);
bodyGroup.add(sunGlow);

const planetObjects = planets.map((data) => {
  const pivot = new THREE.Group();
  pivot.rotation.z = THREE.MathUtils.degToRad(data.inclination);
  bodyGroup.add(pivot);

  const mesh = new THREE.Mesh(
    sphere,
    new THREE.MeshStandardMaterial({
      map: texture(data.file),
      roughness: 0.7,
      metalness: 0.02,
    }),
  );
  mesh.scale.setScalar(data.radius);
  mesh.position.x = data.orbit;
  pivot.add(mesh);

  const points = [];
  for (let i = 0; i <= 160; i += 1) {
    const a = (i / 160) * Math.PI * 2;
    points.push(new THREE.Vector3(Math.cos(a) * data.orbit, 0, Math.sin(a) * data.orbit));
  }
  const orbit = new THREE.Line(new THREE.BufferGeometry().setFromPoints(points), orbitMaterial);
  orbit.rotation.z = pivot.rotation.z;
  bodyGroup.add(orbit);

  return { ...data, pivot, mesh };
});

const earth = planetObjects.find((planet) => planet.name === "Earth");
const moon = new THREE.Mesh(
  new THREE.SphereGeometry(0.62, 36, 18),
  new THREE.MeshStandardMaterial({ map: texture("moon.jpg"), roughness: 0.8 }),
);
earth.mesh.add(moon);

const clouds = new THREE.Mesh(
  new THREE.SphereGeometry(1.055, 72, 36),
  new THREE.MeshStandardMaterial({
    map: texture("earth_clouds.png"),
    transparent: true,
    opacity: 0.36,
    depthWrite: false,
  }),
);
earth.mesh.add(clouds);

const saturn = planetObjects.find((planet) => planet.name === "Saturn");
const ring = new THREE.Mesh(
  new THREE.RingGeometry(1.35, 2.25, 128),
  new THREE.MeshStandardMaterial({
    map: texture("saturn_ring.png"),
    transparent: true,
    opacity: 0.84,
    side: THREE.DoubleSide,
    depthWrite: false,
  }),
);
ring.rotation.x = Math.PI / 2.25;
saturn.mesh.add(ring);

const asteroidGeometry = new THREE.BufferGeometry();
const asteroidPositions = [];
for (let i = 0; i < 1400; i += 1) {
  const angle = Math.random() * Math.PI * 2;
  const radius = THREE.MathUtils.randFloat(39, 45);
  asteroidPositions.push(Math.cos(angle) * radius, THREE.MathUtils.randFloatSpread(2.4), Math.sin(angle) * radius);
}
asteroidGeometry.setAttribute("position", new THREE.Float32BufferAttribute(asteroidPositions, 3));
bodyGroup.add(new THREE.Points(asteroidGeometry, new THREE.PointsMaterial({ color: 0xa7a09a, size: 0.34 })));

let selectedIndex = 2;
let paused = false;
let speed = 1;
let follow = false;
let cinematic = false;
const clock = new THREE.Clock();

function selectedPlanet() {
  return planetObjects[selectedIndex];
}

function updateHud() {
  const planet = selectedPlanet();
  const mode = cinematic ? "Cinematic" : follow ? "Follow" : "Free";
  modeEl.textContent = mode;
  selectedEl.textContent = planet.name;
  speedEl.textContent = `${speed.toFixed(2)}x`;
  planetNameEl.textContent = planet.name;
  planetStatsEl.textContent = `Orbit radius ${planet.orbit.toFixed(1)} · Rotation ${planet.rotation.toFixed(1)}`;
  playButton.textContent = paused ? "▶" : "Ⅱ";
  followButton.classList.toggle("active", follow);
  cinematicButton.classList.toggle("active", cinematic);
}

function select(offset) {
  selectedIndex = (selectedIndex + offset + planetObjects.length) % planetObjects.length;
  updateHud();
}

document.querySelector("#prev").addEventListener("click", () => select(-1));
document.querySelector("#next").addEventListener("click", () => select(1));
playButton.addEventListener("click", () => {
  paused = !paused;
  updateHud();
});
followButton.addEventListener("click", () => {
  follow = !follow;
  cinematic = false;
  updateHud();
});
cinematicButton.addEventListener("click", () => {
  cinematic = !cinematic;
  follow = false;
  updateHud();
});
speedRange.addEventListener("input", () => {
  speed = Number(speedRange.value);
  updateHud();
});

window.addEventListener("keydown", (event) => {
  if (event.key === "Tab") {
    event.preventDefault();
    select(1);
  } else if (event.key === " ") {
    paused = !paused;
    updateHud();
  } else if (event.key.toLowerCase() === "f") {
    follow = !follow;
    cinematic = false;
    updateHud();
  } else if (event.key.toLowerCase() === "c") {
    cinematic = !cinematic;
    follow = false;
    updateHud();
  }
});

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

function animate() {
  const dt = Math.min(clock.getDelta(), 0.05);
  const elapsed = clock.elapsedTime;

  if (!paused) {
    sun.rotation.y += dt * 0.22 * speed;
    sunGlow.rotation.y -= dt * 0.1 * speed;
    planetObjects.forEach((planet) => {
      planet.pivot.rotation.y += dt * planet.speed * 0.26 * speed;
      planet.mesh.rotation.y += dt * planet.rotation * 0.55 * speed;
    });
    clouds.rotation.y += dt * 0.24 * speed;
    moon.position.set(Math.cos(elapsed * 1.8 * speed) * 4.2, 0, Math.sin(elapsed * 1.8 * speed) * 4.2);
    moon.rotation.y += dt * 1.4 * speed;
  }

  const target = new THREE.Vector3();
  selectedPlanet().mesh.getWorldPosition(target);
  if (follow) {
    const desired = target.clone().add(new THREE.Vector3(0, selectedPlanet().radius * 5 + 8, selectedPlanet().radius * 12 + 18));
    camera.position.lerp(desired, 0.055);
    controls.target.lerp(target, 0.09);
  } else if (cinematic) {
    camera.position.set(Math.cos(elapsed * 0.13) * 155, 72 + Math.sin(elapsed * 0.09) * 20, Math.sin(elapsed * 0.13) * 155);
    controls.target.lerp(new THREE.Vector3(0, 0, 0), 0.08);
  }

  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}

updateHud();
animate();
