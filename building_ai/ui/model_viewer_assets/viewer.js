import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

let renderer, camera, controls, group, bridge, selected, bounds, model;
const scene = new THREE.Scene();
scene.background = new THREE.Color('#f4f7fc');
const colors = {Wall:0x82aadd, Floor:0xb2bbc7, RoofCeiling:0xe0bc91,
  FixedWindow:0x55bdab, OperableWindow:0x55bdab, Door:0xb399d4, GlassDoor:0x55bdab};
function choose(id) {
  if (selected) selected.material.emissive.setHex(0);
  selected = group?.children.find(x=>x.isMesh && x.userData.id===id);
  if (selected) selected.material.emissive.setHex(0x805300);
}
function reset() {
  if (!bounds) return;
  const size = Math.max(...bounds.max.toArray().map((v,i)=>v-bounds.min.toArray()[i]),1);
  controls.target.set(0,0,0);
  camera.position.set(size*1.25,-size*1.4,size*1.1);
  camera.near=size/10000; camera.far=size*1000; camera.updateProjectionMatrix(); controls.update();
}
function load(payload) {
  model=payload;
  if(group){scene.remove(group); group.traverse(x=>{x.geometry?.dispose();x.material?.dispose();});}
  group=new THREE.Group(); selected=null;
  bounds=new THREE.Box3(new THREE.Vector3(...payload.bounds[0]),new THREE.Vector3(...payload.bounds[1]));
  const center=bounds.getCenter(new THREE.Vector3());
  for(const element of payload.elements){
    const vertices=element.vertices.map(p=>new THREE.Vector3(...p).sub(center));
    const normal=new THREE.Vector3();
    for(let i=0;i<vertices.length;i++) normal.add(new THREE.Vector3().crossVectors(vertices[i],vertices[(i+1)%vertices.length]));
    const axis=normal.toArray().map(Math.abs).indexOf(Math.max(...normal.toArray().map(Math.abs)));
    const flat=vertices.map(v=>{const a=v.toArray().filter((_,i)=>i!==axis); return new THREE.Vector2(...a);});
    const faces=THREE.ShapeUtils.triangulateShape(flat,[]);
    const geometry=new THREE.BufferGeometry().setFromPoints(vertices);
    geometry.setIndex(faces.flat()); geometry.computeVertexNormals();
    const opening=element.source_type==='OS:SubSurface';
    const material=new THREE.MeshStandardMaterial({color:colors[element.category]||0xa7b5c8,
      side:THREE.DoubleSide,roughness:.85,metalness:0,polygonOffset:opening,polygonOffsetFactor:-2,polygonOffsetUnits:-2});
    const mesh=new THREE.Mesh(geometry,material);mesh.userData={id:element.object_id,category:element.category,opening};
    const lines=new THREE.LineSegments(new THREE.EdgesGeometry(geometry),new THREE.LineBasicMaterial({color:0x304560}));
    mesh.add(lines); group.add(mesh);
  }
  scene.add(group); reset();
  window.viewerState={ready:true,elements:payload.elements.length,unit:payload.unit};
}
window.viewer={load,reset,select:choose,
  showCategory:(category,visible)=>group?.children.forEach(x=>{if(x.userData.category===category)x.visible=visible;}),
  state:()=>({elements:group?.children.length||0,visible:group?.children.filter(x=>x.visible).length||0,
    selected:selected?.userData.id||null,position:camera?.position.toArray(),target:controls?.target.toArray(),unit:model?.unit}),
  // Projection is useful for accessibility/testing; selection still uses actual raycasting.
  screenPoint:(id)=>{
    const mesh=group?.children.find(x=>x.userData.id===id);if(!mesh)return null;
    mesh.geometry.computeBoundingBox();const p=mesh.geometry.boundingBox.getCenter(new THREE.Vector3()).project(camera);
    return [(p.x+1)*innerWidth/2,(1-p.y)*innerHeight/2];
  }
};
new QWebChannel(qt.webChannelTransport,channel=>{
  bridge=channel.objects.bridge;
  try {
    renderer=new THREE.WebGLRenderer({antialias:true,preserveDrawingBuffer:true});
    renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.setSize(innerWidth,innerHeight);
    document.body.appendChild(renderer.domElement);
    camera=new THREE.PerspectiveCamera(45,innerWidth/innerHeight,.01,10000);camera.up.set(0,0,1);
    controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=false;
    scene.add(new THREE.HemisphereLight(0xffffff,0x657383,2.8));
    const light=new THREE.DirectionalLight(0xffffff,2);light.position.set(10,-20,30);scene.add(light);
    const ray=new THREE.Raycaster(), mouse=new THREE.Vector2();let down;
    renderer.domElement.addEventListener('pointerdown',e=>down=[e.clientX,e.clientY]);
    renderer.domElement.addEventListener('pointerup',e=>{
      if(!down||Math.hypot(e.clientX-down[0],e.clientY-down[1])>5||e.button!==0||!group)return;
      mouse.set(e.clientX/innerWidth*2-1,1-e.clientY/innerHeight*2);ray.setFromCamera(mouse,camera);
      // Opening polygons lie on their parent surface. Match the visible overlay
      // when both intersect at the same depth; never invent an opening cut.
      const hits=ray.intersectObjects(group.children,false).filter(x=>x.object.visible);
      hits.sort((a,b)=>Math.abs(a.distance-b.distance)<1e-7
        ? Number(b.object.userData.opening)-Number(a.object.userData.opening)
        : a.distance-b.distance);
      const hit=hits[0];
      if(hit){choose(hit.object.userData.id);bridge.selected(hit.object.userData.id);}
    });
    addEventListener('resize',()=>{camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight);});
    renderer.setAnimationLoop(()=>renderer.render(scene,camera));
    bridge.ready();
  } catch(error) {bridge.failed(String(error));}
});
