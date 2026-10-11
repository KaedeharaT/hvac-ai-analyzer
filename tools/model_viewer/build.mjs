import { build } from 'esbuild';
import { copyFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { resolve, dirname } from 'node:path';
const here=dirname(fileURLToPath(import.meta.url));
const assets=resolve(here,'../../building_ai/ui/model_viewer_assets');
await build({entryPoints:[resolve(assets,'viewer.js')],outfile:resolve(assets,'viewer.bundle.js'),
  bundle:true,minify:true,format:'iife',target:'chrome87',
  nodePaths:[resolve(here,'node_modules')]});
copyFileSync(resolve(here,'node_modules/three/LICENSE'),resolve(assets,'THREE-LICENSE.txt'));
