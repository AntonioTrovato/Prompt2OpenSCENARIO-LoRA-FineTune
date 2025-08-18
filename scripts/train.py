import glob
import os, json, argparse, random
from typing import Dict, List, Any
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
import torch
import xml.etree.ElementTree as ET
from lxml import etree
import yaml

import xml.etree.ElementTree as ET
from typing import Optional, Dict, Any
import re


def reduce_xosc(input_xosc_content: str) -> str:
    """
    Drastically reduces an OpenSCENARIO (.xosc) file to contain ONLY essential elements:
    - Map (RoadNetwork)
    - Time of day
    - Weather
    - Speed limit
    - Entities (at least one with "ego_vehicle" property)
    - Initial positions
    - Events
    - Notes

    Args:
        input_xosc_path: Path to the input .xosc file
        output_xosc_path: Optional path for output file. If None, returns XML string

    Returns:
        XML string of the drastically reduced scenario
    """

    # Parse the input XML
    root = ET.fromstring(input_xosc_content)

    # Create new root element with correct tag name
    new_root = ET.Element("OpenScenario")

    # 1. Minimal FileHeader (required by schema)
    header = ET.SubElement(new_root, "FileHeader")
    header.set("revMajor", "1")
    header.set("revMinor", "0")
    header.set("date", "2024-01-01T00:00:00")
    header.set("description", "Reduced Scenario")
    header.set("author", "Reducer")

    # 2. Minimal CatalogLocations (required by schema)
    ET.SubElement(new_root, "CatalogLocations")

    # 3. Map - Extract RoadNetwork
    road_network = root.find("RoadNetwork")
    if road_network is not None:
        new_root.append(_copy_element(road_network))
    else:
        # Minimal road network
        rn = ET.SubElement(new_root, "RoadNetwork")
        logic_file = ET.SubElement(rn, "LogicFile")
        logic_file.set("filepath", "map.xodr")

    # 4. Entities - Keep only essential ones, ensure ego exists
    entities = root.find("Entities")
    new_entities = ET.SubElement(new_root, "Entities")

    ego_found = False
    if entities is not None:
        for scenario_obj in entities.findall("ScenarioObject"):
            if _is_ego_vehicle(scenario_obj) or _is_essential_entity(scenario_obj):
                new_entities.append(_copy_element(scenario_obj))
                if _is_ego_vehicle(scenario_obj):
                    ego_found = True

    # Create ego if not found
    if not ego_found:
        ego_obj = _create_minimal_ego_vehicle()
        new_entities.append(ego_obj)

    # 5. Minimal Storyboard with only essentials
    storyboard = ET.SubElement(new_root, "Storyboard")

    # Init section
    init = ET.SubElement(storyboard, "Init")
    init_actions = ET.SubElement(init, "Actions")

    # Environment with time of day, weather, speed limit
    _add_minimal_environment(init_actions)

    # Initial positions
    _add_initial_positions(init_actions, new_entities)

    # Extract and add only essential events
    _add_essential_events(storyboard, root)

    # Simple stop trigger
    _add_stop_trigger(storyboard)

    # Format XML
    _indent_xml(new_root)
    xml_string = ET.tostring(new_root, encoding='unicode')
    xml_string = '<?xml version="1.0" encoding="UTF-8"?>\n' + xml_string

    return xml_string


def _copy_element(element: ET.Element) -> ET.Element:
    """Deep copy an XML element"""
    new_elem = ET.Element(element.tag, element.attrib)
    new_elem.text = element.text
    new_elem.tail = element.tail

    for child in element:
        new_elem.append(_copy_element(child))

    return new_elem


def _is_ego_vehicle(scenario_obj: ET.Element) -> bool:
    """Check if scenario object is ego vehicle"""
    name = scenario_obj.get("name", "").lower()
    if "ego" in name:
        return True

    # Check properties
    properties = scenario_obj.find(".//Properties")
    if properties is not None:
        for prop in properties.findall("Property"):
            prop_name = prop.get("name", "").lower()
            prop_value = prop.get("value", "").lower()
            if "ego" in prop_name or "ego" in prop_value:
                return True

    return False


def _is_essential_entity(scenario_obj: ET.Element) -> bool:
    """Check if entity is essential to keep"""
    name = scenario_obj.get("name", "").lower()
    # Keep entities that might be important for events
    essential_keywords = ["target", "leader", "follower", "obstacle"]
    return any(keyword in name for keyword in essential_keywords)


def _create_minimal_ego_vehicle() -> ET.Element:
    """Create minimal ego vehicle"""
    ego_obj = ET.Element("ScenarioObject")
    ego_obj.set("name", "ego_vehicle")

    vehicle = ET.SubElement(ego_obj, "Vehicle")
    vehicle.set("name", "ego_vehicle")
    vehicle.set("vehicleCategory", "car")

    # Minimal bounding box
    bbox = ET.SubElement(vehicle, "BoundingBox")
    center = ET.SubElement(bbox, "Center")
    center.set("x", "1.5")
    center.set("y", "0.0")
    center.set("z", "0.9")
    dimensions = ET.SubElement(bbox, "Dimensions")
    dimensions.set("width", "2.0")
    dimensions.set("length", "5.0")
    dimensions.set("height", "1.8")

    # Minimal performance
    performance = ET.SubElement(vehicle, "Performance")
    performance.set("maxSpeed", "50.0")
    performance.set("maxAcceleration", "10.0")
    performance.set("maxDeceleration", "10.0")

    # Minimal axles
    axles = ET.SubElement(vehicle, "Axles")
    front_axle = ET.SubElement(axles, "FrontAxle")
    front_axle.set("maxSteering", "0.5")
    front_axle.set("wheelDiameter", "0.6")
    front_axle.set("trackWidth", "1.8")
    front_axle.set("positionX", "3.1")
    front_axle.set("positionZ", "0.3")

    rear_axle = ET.SubElement(axles, "RearAxle")
    rear_axle.set("maxSteering", "0.0")
    rear_axle.set("wheelDiameter", "0.6")
    rear_axle.set("trackWidth", "1.8")
    rear_axle.set("positionX", "0.0")
    rear_axle.set("positionZ", "0.3")

    # Properties with ego marker
    properties = ET.SubElement(vehicle, "Properties")
    ego_prop = ET.SubElement(properties, "Property")
    ego_prop.set("name", "ego_vehicle")
    ego_prop.set("value", "true")

    return ego_obj


def _add_minimal_environment(init_actions: ET.Element):
    """Add minimal environment with time of day, weather, speed limit"""
    global_action = ET.SubElement(init_actions, "GlobalAction")
    env_action = ET.SubElement(global_action, "EnvironmentAction")
    environment = ET.SubElement(env_action, "Environment")
    environment.set("name", "Environment")

    # Time of day
    time_of_day = ET.SubElement(environment, "TimeOfDay")
    time_of_day.set("animation", "false")
    time_of_day.set("dateTime", "2024-07-15T12:00:00")

    # Weather
    weather = ET.SubElement(environment, "Weather")
    weather.set("cloudState", "free")

    sun = ET.SubElement(weather, "Sun")
    sun.set("intensity", "1.0")
    sun.set("azimuth", "0.0")
    sun.set("elevation", "1.31")

    fog = ET.SubElement(weather, "Fog")
    fog.set("visualRange", "100000.0")

    precipitation = ET.SubElement(weather, "Precipitation")
    precipitation.set("precipitationType", "dry")
    precipitation.set("intensity", "0.0")

    # Road condition with speed limit
    road_condition = ET.SubElement(environment, "RoadCondition")
    road_condition.set("frictionScaleFactor", "1.0")
    road_props = ET.SubElement(road_condition, "Properties")
    speed_limit_prop = ET.SubElement(road_props, "Property")
    speed_limit_prop.set("name", "speedLimit")
    speed_limit_prop.set("value", "50.0")

    # Notes as property
    notes_prop = ET.SubElement(road_props, "Property")
    notes_prop.set("name", "notes")
    notes_prop.set("value", "Reduced scenario - essential elements only")


def _add_initial_positions(init_actions: ET.Element, entities: ET.Element):
    """Add initial positions for all entities"""
    for i, scenario_obj in enumerate(entities.findall("ScenarioObject")):
        private = ET.SubElement(init_actions, "Private")
        private.set("entityRef", scenario_obj.get("name"))

        private_action = ET.SubElement(private, "PrivateAction")
        teleport = ET.SubElement(private_action, "TeleportAction")
        position = ET.SubElement(teleport, "Position")

        world_pos = ET.SubElement(position, "WorldPosition")
        world_pos.set("x", str(i * 5.0))  # Spread entities apart
        world_pos.set("y", "0.0")
        world_pos.set("z", "0.0")
        world_pos.set("h", "0.0")
        world_pos.set("p", "0.0")
        world_pos.set("r", "0.0")


def _add_essential_events(storyboard: ET.Element, original_root: ET.Element):
    """Extract and add only essential events"""
    story = ET.SubElement(storyboard, "Story")
    story.set("name", "MainStory")

    act = ET.SubElement(story, "Act")
    act.set("name", "MainAct")

    maneuver_group = ET.SubElement(act, "ManeuverGroup")
    maneuver_group.set("maximumExecutionCount", "1")
    maneuver_group.set("name", "MainManeuverGroup")

    actors = ET.SubElement(maneuver_group, "Actors")
    actors.set("selectTriggeringEntities", "false")
    entity_ref = ET.SubElement(actors, "EntityRef")
    entity_ref.set("entityRef", "ego_vehicle")

    # Look for existing events in original
    existing_events = original_root.findall(".//Event")
    if existing_events:
        # Copy first few essential events
        maneuver = ET.SubElement(maneuver_group, "Maneuver")
        maneuver.set("name", "EssentialManeuver")

        for i, event in enumerate(existing_events[:3]):  # Limit to first 3 events
            new_event = _copy_element(event)
            # Ensure event has required attributes
            if not new_event.get("priority"):
                new_event.set("priority", "overwrite")
            if not new_event.get("name"):
                new_event.set("name", f"Event_{i}")
            maneuver.append(new_event)
    else:
        # Create minimal default event
        maneuver = ET.SubElement(maneuver_group, "Maneuver")
        maneuver.set("name", "DefaultManeuver")

        event = ET.SubElement(maneuver, "Event")
        event.set("name", "DefaultEvent")
        event.set("priority", "overwrite")

        # Simple speed action
        action = ET.SubElement(event, "Action")
        action.set("name", "SpeedAction")
        private_action = ET.SubElement(action, "PrivateAction")
        long_action = ET.SubElement(private_action, "LongitudinalAction")
        speed_action = ET.SubElement(long_action, "SpeedAction")

        dynamics = ET.SubElement(speed_action, "SpeedActionDynamics")
        dynamics.set("dynamicsShape", "step")
        dynamics.set("value", "1.0")
        dynamics.set("dynamicsDimension", "time")

        target = ET.SubElement(speed_action, "SpeedActionTarget")
        abs_speed = ET.SubElement(target, "AbsoluteTargetSpeed")
        abs_speed.set("value", "30.0")

        # Start trigger
        start_trigger = ET.SubElement(event, "StartTrigger")
        cond_group = ET.SubElement(start_trigger, "ConditionGroup")
        condition = ET.SubElement(cond_group, "Condition")
        condition.set("name", "StartCondition")
        condition.set("delay", "0")
        condition.set("conditionEdge", "rising")

        by_value = ET.SubElement(condition, "ByValueCondition")
        sim_time = ET.SubElement(by_value, "SimulationTimeCondition")
        sim_time.set("value", "0.0")
        sim_time.set("rule", "greaterThan")

    # Act start trigger
    act_start_trigger = ET.SubElement(act, "StartTrigger")
    act_cond_group = ET.SubElement(act_start_trigger, "ConditionGroup")
    act_condition = ET.SubElement(act_cond_group, "Condition")
    act_condition.set("name", "ActStart")
    act_condition.set("delay", "0")
    act_condition.set("conditionEdge", "rising")

    act_by_value = ET.SubElement(act_condition, "ByValueCondition")
    act_sim_time = ET.SubElement(act_by_value, "SimulationTimeCondition")
    act_sim_time.set("value", "0.0")
    act_sim_time.set("rule", "greaterThan")


def _add_stop_trigger(storyboard: ET.Element):
    """Add simple stop trigger"""
    stop_trigger = ET.SubElement(storyboard, "StopTrigger")
    cond_group = ET.SubElement(stop_trigger, "ConditionGroup")
    condition = ET.SubElement(cond_group, "Condition")
    condition.set("name", "StopCondition")
    condition.set("delay", "0")
    condition.set("conditionEdge", "rising")

    by_value = ET.SubElement(condition, "ByValueCondition")
    sim_time = ET.SubElement(by_value, "SimulationTimeCondition")
    sim_time.set("value", "60.0")
    sim_time.set("rule", "greaterThan")


def _indent_xml(elem: ET.Element, level: int = 0):
    """Add pretty-printing indentation to XML"""
    indent = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = indent
        for child in elem:
            _indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = indent
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = indent


# ---------------------------
# Feature extractor (fornita)
# ---------------------------
def extract_features_from_xosc(xosc_text: str) -> dict:
    feat = {
        "map": None,
        "time_of_day": None,
        "weather": {"cloud_state": None, "precipitation": None, "fog": None, "wind": None},
        "speed_limits": [],
        "entities": [],
        "initial_positions": [],
        "events": [],
        "notes": []
    }
    try:
        root = ET.fromstring(xosc_text)
    except Exception:
        feat["notes"].append("xml_parse_failed")
        return feat

    rn = root.find(".//RoadNetwork/LogicFile")
    if rn is not None:
        feat["map"] = rn.attrib.get("filepath") or rn.attrib.get("file") or None

    tod = root.find(".//Environment/TimeOfDay")
    if tod is not None:
        feat["time_of_day"] = tod.attrib.get("dateTime") or tod.attrib.get("animation") or None

    w = root.find(".//Environment/Weather")
    if w is not None:
        feat["weather"]["cloud_state"] = w.attrib.get("cloudState")
        feat["weather"]["precipitation"] = w.attrib.get("precipitationType") or w.attrib.get("precipitation")
        fog = w.find(".//Fog")
        if fog is not None:
            feat["weather"]["fog"] = fog.attrib.get("visualRange")
        wind = w.find(".//Wind")
        if wind is not None:
            feat["weather"]["wind"] = wind.attrib.get("direction") or wind.attrib.get("speed")

    for sl in root.findall(".//SpeedLimitAction") + root.findall(".//SpeedAction"):
        maxkph = sl.attrib.get("max") or sl.attrib.get("target")
        if maxkph:
            feat["speed_limits"].append(maxkph)

    for ent in root.findall(".//Entities/*"):
        tag = ent.tag.lower()
        name = ent.attrib.get("name") or ent.attrib.get("nameRef")
        etype = "vehicle"
        if "pedestrian" in tag:
            etype = "pedestrian"
        elif "misc" in tag:
            etype = "misc"
        if name:
            feat["entities"].append({"name": name, "type": etype})

    for priv in root.findall(".//Init/Actions/Private"):
        name = priv.attrib.get("entityRef")
        wp = priv.find(".//WorldPosition")
        if wp is not None:
            feat["initial_positions"].append({
                "entity": name,
                "x": wp.attrib.get("x"), "y": wp.attrib.get("y"),
                "z": wp.attrib.get("z"), "h": wp.attrib.get("h")
            })

    for ev in root.findall(".//Storyboard//Event"):
        ev_name = ev.attrib.get("name")
        act = ev.find(".//Action")
        trig = ev.find(".//StartTrigger") or ev.find(".//ConditionGroup")
        desc = []
        if ev_name: desc.append(ev_name)
        if trig is not None:
            for cond in trig.findall(".//ByValueCondition/SimulationTimeCondition"):
                delay = cond.attrib.get("value")
                if delay:
                    desc.append(f"after {delay}s")
        if act is not None:
            atag = next((c.tag for c in list(act) if isinstance(c.tag, str)), None)
            if atag:
                desc.append(atag)
        if desc:
            feat["events"].append(" ".join(desc))
    return feat

# ---------------------------------
# Riduzione on-the-fly dell'XML gold
# ---------------------------------
def minify_xml(x: str) -> str:
    try:
        #parser = etree.XMLParser(remove_blank_text=True, remove_comments=True)
        #root = etree.fromstring(x.encode("utf-8"), parser=parser)
        #return etree.tostring(root, encoding="unicode", pretty_print=False)
        return reduce_xosc(x)
    except Exception:
        return x

from lxml import etree

def build_minimal_xosc_from_features(feat: dict) -> str:
    # Fallbacks
    map_path = feat.get("map") or "Town01.xodr"
    time_of_day = feat.get("time_of_day") or "2025-01-01T12:00:00"
    weather = feat.get("weather") or {}
    cloud = weather.get("cloud_state") or "free"
    precip = weather.get("precipitation") or "dry"

    entities = feat.get("entities") or []
    if not entities:
        entities = [{"name": "ego", "type": "vehicle"}]

    init_pos_by_entity = {}
    for ip in (feat.get("initial_positions") or []):
        init_pos_by_entity[ip.get("entity")] = {
            "x": ip.get("x") or "0", "y": ip.get("y") or "0",
            "z": ip.get("z") or "0", "h": ip.get("h") or "0"
        }

    def mk_pos(name):
        p = init_pos_by_entity.get(name) or {"x":"0","y":"0","z":"0","h":"0"}
        return p["x"], p["y"], p["z"], p["h"]

    E = etree.Element
    root = E("OpenScenario")

    # FileHeader
    fh = E("FileHeader", revMajor="1", revMinor="0",
           date="2025-01-01T00:00:00", description="Minimal scenario")
    root.append(fh)

    # RoadNetwork
    rn = E("RoadNetwork")
    rn.append(E("LogicFile", filepath=map_path))
    root.append(rn)

    # Empty but required
    root.append(E("ParameterDeclarations"))
    root.append(E("CatalogLocations"))

    # Entities
    ents = E("Entities")
    for ent in entities:
        nm = ent.get("name") or "ego"
        typ = ent.get("type") or "vehicle"
        if typ == "pedestrian":
            obj = E("Pedestrian", name=nm, mass="70", model="ped",
                    pedestrianCategory="pedestrian")
            bb = E("BoundingBox")
            bb.append(E("Center", x="0", y="0", z="0.9"))
            bb.append(E("Dimensions", width="0.5", length="0.5", height="1.8"))
            obj.append(bb)
        elif typ == "misc":
            obj = E("MiscObject", name=nm, mass="100",
                    miscObjectCategory="obstacle")
        else:
            obj = E("Vehicle", name=nm, vehicleCategory="car")
            bb = E("BoundingBox")
            bb.append(E("Center", x="0", y="0", z="0.9"))
            bb.append(E("Dimensions", width="2.0", length="4.5", height="1.5"))
            obj.append(bb)
        ents.append(obj)
    root.append(ents)

    # Storyboard
    sb = E("Storyboard")

    # Init: position + environment
    init = E("Init")
    actions = E("Actions")

    # Private Actions: teleport each entity
    for ent in entities:
        nm = ent.get("name") or "ego"
        priv = E("Private", entityRef=nm)
        actions_priv = E("Actions")
        act = E("Action")
        tp = E("TeleportAction")
        pos = E("Position")
        x,y,z,h = mk_pos(nm)
        pos.append(E("WorldPosition", x=x, y=y, z=z, h=h))
        tp.append(pos)
        act.append(tp)
        actions_priv.append(act)
        priv.append(actions_priv)
        actions.append(priv)

    # GlobalAction: Environment
    gact = E("GlobalAction")
    envact = E("EnvironmentAction")
    env = E("Environment", name="default_env")
    env.append(E("TimeOfDay", dateTime=time_of_day))
    env.append(E("Weather", cloudState=cloud, precipitationType=precip))
    envact.append(env)
    gact.append(envact)
    actions.append(gact)

    init.append(actions)
    sb.append(init)

    # Main Story
    story = E("Story", name="main_story")
    act = E("Act", name="act_1")
    for ent in entities:
        nm = ent.get("name") or "ego"
        mg = E("ManeuverGroup", name=f"mg_{nm}")
        man = E("Maneuver", name=f"man_{nm}")
        ev = E("Event", name=f"ev_{nm}", priority="overwrite")
        ev_action = E("Action")
        ev_action.append(E("ControllerAction"))
        ev.append(ev_action)
        # Event StartTrigger
        st = E("StartTrigger")
        cg = E("ConditionGroup")
        c = E("Condition", delay="0", conditionEdge="rising")
        byv = E("ByValueCondition")
        byv.append(E("SimulationTimeCondition", value="0", rule="greaterThan"))
        c.append(byv)
        cg.append(c)
        st.append(cg)
        ev.append(st)
        man.append(ev)
        mg.append(man)
        act.append(mg)

    # Act StartTrigger
    st_act = E("StartTrigger")
    cg_act = E("ConditionGroup")
    c_act = E("Condition", delay="0", conditionEdge="rising")
    byv_act = E("ByValueCondition")
    byv_act.append(E("SimulationTimeCondition", value="0", rule="greaterThan"))
    c_act.append(byv_act)
    cg_act.append(c_act)
    st_act.append(cg_act)
    act.append(st_act)

    story.append(act)
    sb.append(story)
    root.append(sb)

    return etree.tostring(root, encoding="unicode", pretty_print=False)

def reduce_assistant(xosc_text: str, mode: str) -> str:
    if mode == "minify_only":
        return minify_xml(xosc_text)
    if mode == "features_skeleton":
        try:
            feats = extract_features_from_xosc(xosc_text)
            return build_minimal_xosc_from_features(feats)
        except Exception:
            return minify_xml(xosc_text)
    # "none"
    return xosc_text

# -----
# Main
# -----
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default="config/lora-codellama13b.yaml")
    p.add_argument("--jsonl_path", default="")
    p.add_argument("--use_feature_hints", action="store_true")
    p.add_argument("--reduce_mode", choices=["none","minify_only","features_skeleton"], default="minify_only")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.cfg))
    random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])

    tok = AutoTokenizer.from_pretrained(cfg["base_model"], use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # Template single-turn compatibile con assistant_only_loss
    tok.chat_template = (
        "{{ bos_token }}"
        "{% for m in messages %}"
        "{% if m['role'] == 'user' %}"
        "[INST] {{ m['content'] }} [/INST]"
        "{% elif m['role'] == 'assistant' %}"
        "{% generation %}{{ m['content'] }}{% endgeneration %}{{ eos_token }}"
        "{% endif %}"
        "{% endfor %}"
    )

    from transformers import BitsAndBytesConfig

    use_4bit = bool(cfg.get("load_in_4bit", False))
    dtype = torch.bfloat16 if str(cfg.get("torch_dtype", "bfloat16")) == "bfloat16" else torch.float16

    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=cfg.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_compute_dtype=torch.bfloat16 if str(
                cfg.get("bnb_4bit_compute_dtype", "bfloat16")) == "bfloat16" else torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        torch_dtype=dtype if not use_4bit else None,
        quantization_config=bnb_config if use_4bit else None,
        device_map="auto" if use_4bit else None,  # utile con 4-bit
    )
    model.config.pad_token_id = tok.pad_token_id
    if cfg.get("gradient_checkpointing", False):
        model.config.use_cache = False

    # Dataset
    if args.jsonl_path:
        rows = [json.loads(line) for line in open(args.jsonl_path, "r", encoding="utf-8")]
        ds = Dataset.from_list(rows)
    else:
        ds = load_dataset(cfg["hf_dataset_repo"], split="train")

    ds = ds.train_test_split(test_size=cfg.get("val_split_ratio", 0.1), seed=cfg["seed"])
    ds = DatasetDict({"train": ds["train"], "validation": ds["test"]})

    stop_seq = cfg["stop_sequence"]

    peft_cfg = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"],
        bias=cfg["bias"],
        task_type="CAUSAL_LM"
    )

    def to_messages(ex):
        system = ex["system"].strip()
        user = ex["user"].strip()
        assistant = reduce_assistant(ex["assistant"].strip(), args.reduce_mode)
        if not assistant.endswith(stop_seq):
            assistant += stop_seq

        # fondi il system nel primo user (stile CodeLlama)
        user_with_sys = f"<<SYS>>\n{system}\n<</SYS>>\n\n{user}"
        if args.use_feature_hints:
            try:
                feats = extract_features_from_xosc(ex["assistant"])
                user_with_sys += "\n\n<HINTS>\n" + json.dumps(feats, ensure_ascii=False) + "\n</HINTS>\n"
            except Exception:
                pass

        return {
            "messages": [
                {"role": "user", "content": user_with_sys},
                {"role": "assistant", "content": assistant},
            ]
        }

    ds = ds.map(to_messages)

    def _ok_len(batch):
        return [
            len(tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False))
            <= cfg["max_length"]
            for msgs in batch["messages"]
        ]

    ds["train"] = ds["train"].filter(_ok_len, batched=True)

    sft_cfg = SFTConfig(
        max_length=cfg["max_length"],
        packing=cfg["packing"],
        eval_strategy="steps",
        eval_steps=cfg["eval_steps"],
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg["num_epochs"],
        per_device_train_batch_size=cfg["train_batch_size"],
        gradient_accumulation_steps=cfg["grad_accum_steps"],
        per_device_eval_batch_size=cfg["eval_batch_size"],
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=cfg["warmup_ratio"],
        output_dir=cfg["output_dir"],
        report_to=["tensorboard"],
        bf16=(str(cfg.get("torch_dtype", "bfloat16")) == "bfloat16"),
        gradient_checkpointing=cfg["gradient_checkpointing"],
        assistant_only_loss=True,
        save_total_limit=cfg.get("save_total_limit", 3),
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        peft_config=peft_cfg,
        processing_class=tok,
    )

    last_ckpt = None
    if os.path.isdir(cfg["output_dir"]):
        candidates = sorted(glob.glob(os.path.join(cfg["output_dir"], "checkpoint-*")),
                            key=os.path.getmtime)
        if candidates:
            last_ckpt = candidates[-1]

    if last_ckpt:
        print(f"Resuming from checkpoint: {last_ckpt}")
        trainer.train(resume_from_checkpoint=last_ckpt)
    else:
        trainer.train()

    trainer.save_model(cfg["output_dir"])
    tok.save_pretrained(cfg["output_dir"])

if __name__ == "__main__":
    main()
