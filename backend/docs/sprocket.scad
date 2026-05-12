/*
  sprocket.scad
  Project-local sprocket reference library for OpenSCAD generation.
  Based on the inspected Sprockets.scad v1.1 structure by Shawn Steele,
  with Dallen Wilson keyway and set-screw feature behavior preserved in
  standalone form so RAG can reference one authoritative .scad file.

  Main call:
    sprocket(size=40, teeth=17, bore=10/25.4, hub_diameter=22/25.4,
             hub_height=14/25.4, keyway=1, setscrew=1);

  Units:
    size is ANSI chain number.
    bore, hub_diameter, hub_height are inches.
    Internal geometry is converted to millimeters.
*/

FUDGE_BORE = 0.5;
FUDGE_ROLLER = 0;
FUDGE_TEETH = 1;
FUDGE_KEYWAY = 0.5;

function inches2mm(inches) = inches * 25.4;
function mm2inches(mm) = mm / 25.4;

module sprocket(size=25, teeth=9, bore=5/16, hub_diameter=0, hub_height=0, keyway=0, setscrew=0, keyway_threads=1) {
  bore_radius_mm = inches2mm(bore) / 2;
  hub_radius_mm = inches2mm(hub_diameter) / 2;
  hub_height_mm = inches2mm(hub_height);
  sprocket_thickness = get_thickness(size);
  hub_wall_thickness = (hub_diameter - bore) / 2;
  kw_width = inches2mm(get_keyway_width(bore));
  ss_width = get_setscrew_width(bore);
  ss_length_mm = inches2mm(hub_wall_thickness) + 2;

  difference() {
    union() {
      sprocket_plate(size, teeth);
      if (hub_diameter != 0 && hub_height != 0)
        cylinder(h = hub_height_mm, r = hub_radius_mm);
    }

    if (bore != 0) {
      translate([0, 0, -1])
        cylinder(h = hub_height_mm + inches2mm(sprocket_thickness) + 2, r = bore_radius_mm + FUDGE_BORE);

      if (keyway != 0 && kw_width > 0)
        translate([-(bore_radius_mm + FUDGE_BORE + kw_width / 2), -kw_width / 2, -1])
          cube([kw_width + FUDGE_KEYWAY, kw_width + FUDGE_KEYWAY, hub_height_mm + 2], false);

      if (setscrew != 0 && ss_width > 0) {
        if (keyway_threads > 0)
          rotate([0, 90, 0])
            translate([-(inches2mm(sprocket_thickness) + max(0, hub_height_mm - inches2mm(sprocket_thickness)) / 2), 0, -(bore_radius_mm + ss_length_mm - 1)])
              cylinder(h = ss_length_mm, d = inches2mm(ss_width), center = false);

        rotate([90, 0, 0])
          translate([0, inches2mm(sprocket_thickness) + max(0, hub_height_mm - inches2mm(sprocket_thickness)) / 2, bore_radius_mm - 1])
            cylinder(h = ss_length_mm, d = inches2mm(ss_width), center = false);
      }
    }
  }
}

module sprocket_plate(size, teeth) {
  angle = 360 / teeth;
  pitch = inches2mm(get_pitch(size));
  roller = inches2mm(get_roller_diameter(size) / 2);
  thickness = inches2mm(get_thickness(size));
  pitch_radius = inches2mm(get_pitch(size) / sin(180 / teeth)) / 2;
  middle_radius = sqrt(pow(pitch_radius, 2) - pow(pitch / 2, 2));
  fudge_teeth_x = FUDGE_TEETH * cos(angle / 2);
  fudge_teeth_y = FUDGE_TEETH * sin(angle / 2);

  difference() {
    intersection() {
      cylinder(r = pitch_radius - roller + pitch / 2, h = thickness);
      union() {
        for (i = [0 : teeth - 1])
          rotate([0, 0, angle * i])
            intersection() {
              translate([-fudge_teeth_x, pitch_radius - fudge_teeth_y, 0])
                cylinder(r = pitch - roller - FUDGE_ROLLER - FUDGE_TEETH, h = thickness);
              rotate([0, 0, angle])
                translate([fudge_teeth_x, pitch_radius - fudge_teeth_y, 0])
                  cylinder(r = pitch - roller - FUDGE_ROLLER - FUDGE_TEETH, h = thickness);
            }

        for (i = [0 : teeth - 1])
          rotate([0, 0, angle * i - angle / 2])
            translate([-pitch / 2, -0.01, 0])
              cube([pitch, middle_radius + 0.01, thickness]);
      }
    }

    for (i = [0 : teeth - 1])
      rotate([0, 0, angle * i])
        translate([0, pitch_radius, -1])
          cylinder(r = roller + FUDGE_ROLLER, h = thickness + 2);
  }
}

function get_pitch(size) =
  size == 25 ? 1/4 :
  size == 35 ? 3/8 :
  size == 40 ? 1/2 :
  size == 41 ? 1/2 :
  size == 50 ? 5/8 :
  size == 60 ? 3/4 :
  size == 80 ? 1 :
  size == 1 ? 1/2 :
  size == 2 ? 1/2 :
  size == 420 ? 1/2 :
  size == 425 ? 1/2 :
  size == 428 ? 1/2 :
  size == 520 ? 5/8 :
  size == 525 ? 5/8 :
  size == 530 ? 5/8 :
  size == 630 ? 3/4 : 0;

function get_roller_diameter(size) =
  size == 25 ? 0.130 :
  size == 35 ? 0.200 :
  size == 40 ? 5/16 :
  size == 41 ? 0.306 :
  size == 50 ? 0.400 :
  size == 60 ? 15/32 :
  size == 80 ? 5/8 :
  size == 1 ? 5/16 :
  size == 2 ? 5/16 :
  size == 420 ? 5/16 :
  size == 425 ? 5/16 :
  size == 428 ? 0.335 :
  size == 520 ? 0.400 :
  size == 525 ? 0.400 :
  size == 530 ? 0.400 :
  size == 630 ? 15/32 : 0;

function get_thickness(size) =
  size == 25 ? 0.110 :
  size == 35 ? 0.168 :
  size == 40 ? 0.284 :
  size == 41 ? 0.227 :
  size == 50 ? 0.343 :
  size == 60 ? 0.459 :
  size == 80 ? 0.575 :
  size == 1 ? 0.110 :
  size == 2 ? 0.084 :
  size == 420 ? 0.227 :
  size == 425 ? 0.284 :
  size == 428 ? 0.284 :
  size == 520 ? 0.227 :
  size == 525 ? 0.284 :
  size == 530 ? 0.343 :
  size == 630 ? 0.343 : 0;

function get_chainplate_size(size) =
  size == 25 ? 0.228 :
  size == 35 ? 0.346 :
  size == 40 ? 0.469 :
  size == 41 ? 0.390 :
  size == 50 ? 0.585 :
  size == 60 ? 0.709 :
  size == 80 ? 0.949 : 0;

function get_keyway_width(bore) =
  bore <= 0.375 ? 0 :
  bore <= 0.5625 ? 0.125 :
  bore <= 0.875 ? 0.1875 :
  bore <= 1.25 ? 0.250 :
  bore <= 1.375 ? 0.3125 :
  bore <= 1.75 ? 0.375 :
  bore <= 2.25 ? 0.5 :
  bore <= 2.75 ? 0.625 :
  bore <= 3.25 ? 0.75 :
  bore <= 3.75 ? 0.875 :
  bore <= 4.5 ? 1 :
  bore <= 5.5 ? 1.25 :
  bore <= 6.5 ? 1.5 :
  bore <= 7.5 ? 1.75 :
  bore <= 8.9375 ? 2 :
  bore <= 10.9375 ? 2.5 : 0;

function get_setscrew_width(bore) =
  bore <= 0.375 ? 0.0 :
  bore <= 0.5625 ? 0.1875 :
  bore <= 0.875 ? 0.25 :
  bore <= 1.25 ? 0.3125 :
  bore <= 1.75 ? 0.375 :
  bore <= 2.75 ? 0.5 :
  bore <= 3.25 ? 0.625 : 0;

function get_setscrew_threads(ss_width) =
  ss_width <= 0.1875 ? 24 :
  ss_width <= 0.25 ? 20 :
  ss_width <= 0.3125 ? 18 :
  ss_width <= 0.375 ? 16 :
  ss_width <= 0.5 ? 13 :
  ss_width <= 0.625 ? 11 : 0;
