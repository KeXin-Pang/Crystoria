import { createHash } from "node:crypto";

export const UPSTREAM_HASHES = {
  "StructureScene.svelte": "3d44a4a389980cee9ccdcc048151d6c88a30a5b8080c0afa21d92433534bd47c",
  "Bond.svelte": "c5c2ed07d82037b528e15b6411771859ddcf781dc70e24683a54d46b0b2a1ffc",
};

export function publication_is_base_site(site) {
  return typeof site?.properties?.orig_site_idx !== "number";
}

export function publication_in_closed_cell(site) {
  return Array.isArray(site?.abc)
    ? site.abc.every((coordinate) => coordinate >= -1e-7 && coordinate <= 1 + 1e-7)
    : publication_is_base_site(site);
}

export function publication_keep_bond(site_a, site_b) {
  return publication_is_base_site(site_a) || publication_is_base_site(site_b);
}

export function publication_source_bond_visible(structure, bond) {
  const anchor = structure.sites[bond.site_idx_1];
  const target = structure.sites[bond.site_idx_2];
  return publication_is_base_site(anchor) && publication_in_closed_cell(anchor)
    && publication_in_closed_cell({abc: target.abc.map((value, axis) => value + bond.cell_shift[axis])});
}

export function publicationStyle(source, name) {
  if (!(name in UPSTREAM_HASHES)) return source;
  const digest = createHash("sha256").update(source).digest("hex");
  if (digest !== UPSTREAM_HASHES[name]) throw new Error(`Publication source drift: ${name}`);
  if (name === "StructureScene.svelte") {
    source = source.replace('<script lang="ts">', `<script lang="ts">
  const publication_glossy = new URLSearchParams(window.location.search).get("atom_finish") !== "matte";
  const publication_material = publication_glossy
    ? { roughness: 0.16, metalness: 0.08 }
    : { roughness: 1, metalness: 0 };
  window.__CRYSTORIA_ATOM_MATERIAL__ = publication_material;
  const publication_home_cell_only = new URLSearchParams(window.location.search).get("atom_scope") !== "images";
  const publication_source_bonds = JSON.parse(new URLSearchParams(window.location.search).get("manual_bonds") || "null");
  ${publication_is_base_site.toString()}
  ${publication_in_closed_cell.toString()}
  ${publication_keep_bond.toString()}
  ${publication_source_bond_visible.toString()}
  function publication_render_sites(sites) {
    const visible = publication_home_cell_only ? sites.filter((row) => publication_in_closed_cell(row.site)) : sites;
    window.__CRYSTORIA_VISIBLE_ATOMS__ = {
      base_sites: sites.filter((site) => !site.is_image_atom).length,
      image_sites_available_for_bonds: sites.filter((site) => site.is_image_atom).length,
      visible_images: visible.filter((site) => site.is_image_atom).length,
      visible_boundary_images: visible.filter((row) => row.is_image_atom && publication_in_closed_cell(row.site)).length,
      visible_outside_cell: visible.filter((row) => !publication_in_closed_cell(row.site)).length,
      visible_total: visible.length,
    };
    return visible;
  }`);
    source = source.replace("merge_split_partial_sites(structure.sites, hidden_elements)",
      "publication_render_sites(merge_split_partial_sites(structure.sites, hidden_elements))");
    source = source.replace("if (!structure?.sites || bonds_to_render.length === 0) return []",
      "if (!structure?.sites || (!publication_source_bonds?.length && bonds_to_render.length === 0)) return []");
    source = source.replace("for (const bond_data of bonds_to_render) {", `const publication_bonds = publication_source_bonds
      ? publication_source_bonds.filter((bond) => publication_source_bond_visible(structure, bond))
          .map((bond) => structure_bond_to_bond_pair(structure, bond))
      : publication_home_cell_only
        ? bonds_to_render.filter((bond) => publication_keep_bond(structure.sites[bond.site_idx_1], structure.sites[bond.site_idx_2]))
        : bonds_to_render;
    window.__CRYSTORIA_BOND_SCOPE__ = {
      candidate_bonds: publication_source_bonds?.length ?? bonds_to_render.length,
      retained_bonds: publication_bonds.length,
      source_anchored_closed_cell: Boolean(publication_source_bonds),
      image_image_bonds: publication_bonds.filter((bond) => !publication_keep_bond(structure.sites[bond.site_idx_1], structure.sites[bond.site_idx_2])).length,
    };
    for (const bond_data of publication_bonds) {`);
    const start = source.indexOf("<!-- Instanced rendering for full occupancy atoms -->");
    const end = source.indexOf("<!-- Invisible full-sphere hit targets", start);
    if (start < 0 || end < 0) throw new Error("Atom material anchors changed");
    return source.slice(0, start) + source.slice(start, end).replaceAll(
      "<T.MeshStandardMaterial", "<T.MeshStandardMaterial {...publication_material}",
    ) + source.slice(end);
  }
  if (name === "Bond.svelte") {
    return source.replace('<script lang="ts">', `<script lang="ts">
  const publication_split_bonds = new URLSearchParams(window.location.search).get("bond_style") !== "gradient";
  window.__CRYSTORIA_SPLIT_BONDS__ = publication_split_bonds;`)
      .replace("saturation = 0.5, brightness = 0.7", "saturation = 1.0, brightness = 1.0")
      .replace("uniform float brightness;", "uniform float brightness;\n    uniform float splitColors;")
      .replace("mix(vColorStart, vColorEnd, vYPosition + 0.5)",
        "mix(vColorStart, vColorEnd, splitColors > 0.5 ? step(0.0, vYPosition) : vYPosition + 0.5)")
      .replace("base_color * (ambientIntensity + directionalIntensity * diffuse)",
        "base_color * (splitColors > 0.5 ? (0.72 + 0.28 * diffuse) : (ambientIntensity + directionalIntensity * diffuse))")
      .replace("brightness: { value: brightness },", "brightness: { value: brightness },\n        splitColors: { value: publication_split_bonds ? 1 : 0 },")
      .replace("temp_color.set(color).convertSRGBToLinear()", "temp_color.set(color)")
      .replace("const count = group.instances.length", "const count = group.instances.length;\n    window.__CRYSTORIA_RENDERED_BONDS__ = count")
      .replace("group.thickness, group.thickness, 1, 8", "group.thickness, group.thickness, 1, 24");
  }
  return source;
}
