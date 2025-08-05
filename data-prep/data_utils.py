import os
import sys
import numpy as np
import vector
import xml.etree.ElementTree as ET
import gzip

def read_lhe_events(lhe_path, max_events=10):
    print( "reading lhe events" )
    events = []
    with gzip.open(lhe_path, 'rt') as f:  # 'rt' = read text mode
        inside_event = False
        current_event_lines = []
        for line in f:
            if "<event>" in line:
                inside_event = True
                current_event_lines = []
                continue
            elif "</event>" in line:
                inside_event = False
                event = parse_event(current_event_lines)
                events.append(event)
                if len(events) >= max_events:
                    break
            elif inside_event:
                current_event_lines.append(line.strip())
    return events

def parse_event(event_lines):
    n_particles = int(event_lines[0].split()[0])
    particle_lines = event_lines[1:1 + n_particles]
    event_data = []
    final_state_particles = []
    for line in particle_lines:
        parts = line.strip().split()
        pid = int(parts[0])
        status = int(parts[1])
        px, py, pz, E = map(float, parts[6:10])
        if status == 1 or status == -1:
            event_data.append({
                "pid": pid,
                "p": [E, px, py, pz],
                "status": status
            })
    return event_data

def boost_to_com_frame( event, check=False ):
    e4m = vector.obj( px=0.0, py=0.0, pz=0.0, E=0.0 )
    for p in event:
        if p["status"]==1:
            p4m = vector.obj( px=p["p"][1], py=p["p"][2], pz=p["p"][3], E=p["p"][0] )
            e4m = e4m.add( p4m )
    fs4m = vector.obj( px=0.0, py=0.0, pz=0.0, E=0.0 )
    for p in event:
        p4m = vector.obj( px=p["p"][1], py=p["p"][2], pz=p["p"][3], E=p["p"][0] )
        p4m_com = p4m.boostCM_of( e4m )
        fs4m = fs4m.add( p4m_com )
        p["p"][0] = p4m_com.E.item()
        p["p"][1] = p4m_com.px.item()
        p["p"][2] = p4m_com.py.item()
        p["p"][3] = p4m_com.pz.item()
    if check == True:
        fsM2 = fs4m.dot( fs4m )
        init_parton_energy_com = np.sqrt(fsM2)/4
        print( init_parton_energy_com )
    return event

def boost_events_to_com_frame( events, check=False ):
    print( "boosting events to com frame" )
    for i, event in enumerate(events):
        e4m = vector.obj( px=0.0, py=0.0, pz=0.0, E=0.0 )
        for p in event:
            if p["status"]==1:
                p4m = vector.obj( px=p["p"][1], py=p["p"][2], pz=p["p"][3], E=p["p"][0] )
                e4m = e4m.add( p4m )
        fs4m = vector.obj( px=0.0, py=0.0, pz=0.0, E=0.0 )
        for p in event:
            p4m = vector.obj( px=p["p"][1], py=p["p"][2], pz=p["p"][3], E=p["p"][0] )
            p4m_com = p4m.boostCM_of( e4m )
            fs4m = fs4m.add( p4m_com )
            p["p"][0] = p4m_com.E.item()
            p["p"][1] = p4m_com.px.item()
            p["p"][2] = p4m_com.py.item()
            p["p"][3] = p4m_com.pz.item()
        if check == True:
            fsM2 = fs4m.dot( fs4m )
            init_parton_energy_com = np.sqrt(fsM2)/4
            print( init_parton_energy_com )
        if i%10000==0:
            print( "done up to: ", i )
    return events
