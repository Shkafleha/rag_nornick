"""
Генератор синтетических данных для домена АСУТП.
По умолчанию пишет в SQLite (asutp.db). Для PostgreSQL — поменять DB_URL.

Запуск:
    pip install sqlalchemy faker
    python scripts/gen_asutp.py
"""
import random
from datetime import datetime, timedelta
from faker import Faker
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Text,
)
from sqlalchemy.orm import declarative_base, relationship, Session

DB_URL = "sqlite:///asutp.db"
# DB_URL = "postgresql+psycopg2://user:pass@localhost:5432/asutp"

SEED = 42
random.seed(SEED)
fake = Faker("ru_RU")
Faker.seed(SEED)

# --- Объёмы ---
N_PLANTS = 3
UNITS_PER_PLANT = (2, 4)
EQUIP_PER_UNIT = (3, 6)
TAGS_PER_EQUIP = (4, 8)
N_OPERATORS = 30
N_SHIFTS = 60                 # ~20 дней по 3 смены
MEASUREMENTS_PER_TAG = 200    # точек временного ряда на тег
ALARM_RATE = 0.02             # доля измерений, порождающих алярм
MAINT_PER_EQUIP = (0, 3)

Base = declarative_base()


class Plant(Base):
    __tablename__ = "plants"
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    city = Column(String(100))
    commissioned_at = Column(DateTime)
    units = relationship("Unit", back_populates="plant")


class Unit(Base):
    __tablename__ = "units"
    id = Column(Integer, primary_key=True)
    plant_id = Column(Integer, ForeignKey("plants.id"), nullable=False)
    code = Column(String(20), nullable=False)
    name = Column(String(100), nullable=False)
    process_type = Column(String(50))  # ректификация, крекинг, компримирование...
    plant = relationship("Plant", back_populates="units")
    equipment = relationship("Equipment", back_populates="unit")


class Equipment(Base):
    __tablename__ = "equipment"
    id = Column(Integer, primary_key=True)
    unit_id = Column(Integer, ForeignKey("units.id"), nullable=False)
    tag_code = Column(String(30), nullable=False)         # P-101, K-202...
    type = Column(String(50))                             # pump, compressor, reactor, heat_exchanger, column
    manufacturer = Column(String(80))
    model = Column(String(80))
    installed_at = Column(DateTime)
    status = Column(String(20))                           # running, stopped, maintenance
    unit = relationship("Unit", back_populates="equipment")
    tags = relationship("Tag", back_populates="equipment")
    maintenance = relationship("MaintenanceEvent", back_populates="equipment")


class Tag(Base):
    __tablename__ = "tags"
    id = Column(Integer, primary_key=True)
    equipment_id = Column(Integer, ForeignKey("equipment.id"), nullable=False)
    tag_name = Column(String(50), nullable=False)         # TI-101, PI-202, FI-303
    description = Column(String(200))
    parameter = Column(String(30))                        # temperature, pressure, flow, level, vibration
    unit_of_measure = Column(String(20))                  # °C, bar, m3/h, %, mm/s
    lo_lo = Column(Float)
    lo = Column(Float)
    hi = Column(Float)
    hi_hi = Column(Float)
    equipment = relationship("Equipment", back_populates="tags")
    measurements = relationship("Measurement", back_populates="tag")
    alarms = relationship("Alarm", back_populates="tag")


class Measurement(Base):
    __tablename__ = "measurements"
    id = Column(Integer, primary_key=True)
    tag_id = Column(Integer, ForeignKey("tags.id"), nullable=False)
    ts = Column(DateTime, nullable=False)
    value = Column(Float, nullable=False)
    quality = Column(String(20))   # good, bad, uncertain
    tag = relationship("Tag", back_populates="measurements")


class Alarm(Base):
    __tablename__ = "alarms"
    id = Column(Integer, primary_key=True)
    tag_id = Column(Integer, ForeignKey("tags.id"), nullable=False)
    ts = Column(DateTime, nullable=False)
    severity = Column(String(20))   # warning, high, critical
    message = Column(String(300))
    acknowledged_by = Column(Integer, ForeignKey("operators.id"))
    acknowledged_at = Column(DateTime)
    tag = relationship("Tag", back_populates="alarms")


class Operator(Base):
    __tablename__ = "operators"
    id = Column(Integer, primary_key=True)
    full_name = Column(String(120), nullable=False)
    role = Column(String(50))      # operator, senior_operator, shift_supervisor
    hired_at = Column(DateTime)


class Shift(Base):
    __tablename__ = "shifts"
    id = Column(Integer, primary_key=True)
    plant_id = Column(Integer, ForeignKey("plants.id"), nullable=False)
    started_at = Column(DateTime, nullable=False)
    ended_at = Column(DateTime, nullable=False)
    supervisor_id = Column(Integer, ForeignKey("operators.id"))
    notes = Column(Text)


class MaintenanceEvent(Base):
    __tablename__ = "maintenance_events"
    id = Column(Integer, primary_key=True)
    equipment_id = Column(Integer, ForeignKey("equipment.id"), nullable=False)
    started_at = Column(DateTime, nullable=False)
    ended_at = Column(DateTime)
    type = Column(String(30))   # planned, unplanned, inspection
    description = Column(String(300))
    performed_by = Column(String(120))
    equipment = relationship("Equipment", back_populates="maintenance")


# --- Справочники ---
PROCESS_TYPES = ["ректификация", "крекинг", "компримирование", "сепарация",
                 "теплообмен", "осушка", "гидроочистка"]
EQUIP_TYPES = {
    "pump":           ("P",  "Насос"),
    "compressor":     ("K",  "Компрессор"),
    "reactor":        ("R",  "Реактор"),
    "heat_exchanger": ("E",  "Теплообменник"),
    "column":         ("C",  "Колонна"),
    "tank":           ("T",  "Резервуар"),
}
PARAMS = {
    # parameter: (prefix, unit, base, spread, lo_lo, lo, hi, hi_hi)
    "temperature": ("TI", "°C",   80,  15, 20,  40, 130, 160),
    "pressure":    ("PI", "bar",  10,   2,  2,   5,  15,  20),
    "flow":        ("FI", "m3/h", 50,  10,  5,  15,  80, 100),
    "level":       ("LI", "%",    60,  15,  5,  20,  85,  95),
    "vibration":   ("VI", "mm/s",  3, 1.0,  0, 0.5, 7.0, 10.0),
}
MANUFACTURERS = ["Siemens", "ABB", "Emerson", "Yokogawa", "Honeywell", "ОМЗ", "Уралмаш"]


def make_engine_and_schema():
    engine = create_engine(DB_URL, future=True)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    return engine


def generate(session: Session):
    # --- Plants ---
    plants = []
    for i in range(N_PLANTS):
        p = Plant(
            name=f"НПЗ-{i+1} «{fake.last_name()}»",
            city=fake.city(),
            commissioned_at=fake.date_time_between(start_date="-30y", end_date="-5y"),
        )
        plants.append(p)
    session.add_all(plants)
    session.flush()

    # --- Operators ---
    operators = []
    for _ in range(N_OPERATORS):
        operators.append(Operator(
            full_name=fake.name(),
            role=random.choices(
                ["operator", "senior_operator", "shift_supervisor"],
                weights=[7, 2, 1])[0],
            hired_at=fake.date_time_between(start_date="-15y", end_date="-1y"),
        ))
    session.add_all(operators)
    session.flush()

    # --- Units / Equipment / Tags ---
    units, equipments, tags = [], [], []
    for plant in plants:
        for u_idx in range(random.randint(*UNITS_PER_PLANT)):
            unit = Unit(
                plant_id=plant.id,
                code=f"У-{plant.id}{u_idx+1:02d}",
                name=f"Установка {random.choice(PROCESS_TYPES)}",
                process_type=random.choice(PROCESS_TYPES),
            )
            session.add(unit); session.flush()
            units.append(unit)

            for e_idx in range(random.randint(*EQUIP_PER_UNIT)):
                etype = random.choice(list(EQUIP_TYPES.keys()))
                prefix, _ = EQUIP_TYPES[etype]
                eq = Equipment(
                    unit_id=unit.id,
                    tag_code=f"{prefix}-{unit.id}{e_idx+1:02d}",
                    type=etype,
                    manufacturer=random.choice(MANUFACTURERS),
                    model=fake.bothify(text="MDL-####"),
                    installed_at=fake.date_time_between(start_date="-20y", end_date="-1y"),
                    status=random.choices(
                        ["running", "stopped", "maintenance"],
                        weights=[8, 1, 1])[0],
                )
                session.add(eq); session.flush()
                equipments.append(eq)

                # Tags для оборудования
                params_pool = random.sample(list(PARAMS.keys()),
                                            k=min(len(PARAMS), random.randint(*TAGS_PER_EQUIP)))
                for t_idx, param in enumerate(params_pool):
                    pfx, uom, base, spread, lolo, lo, hi, hihi = PARAMS[param]
                    tag = Tag(
                        equipment_id=eq.id,
                        tag_name=f"{pfx}-{eq.id}{t_idx+1:02d}",
                        description=f"{param} на {eq.tag_code}",
                        parameter=param,
                        unit_of_measure=uom,
                        lo_lo=lolo, lo=lo, hi=hi, hi_hi=hihi,
                    )
                    session.add(tag); session.flush()
                    tags.append((tag, base, spread))

    # --- Shifts ---
    shift_starts = []
    base_date = datetime.now() - timedelta(days=N_SHIFTS // 3)
    for i in range(N_SHIFTS):
        start = base_date + timedelta(hours=8 * i)
        end = start + timedelta(hours=8)
        plant = random.choice(plants)
        sup = random.choice([o for o in operators if o.role == "shift_supervisor"] or operators)
        session.add(Shift(
            plant_id=plant.id,
            started_at=start, ended_at=end,
            supervisor_id=sup.id,
            notes=fake.sentence(nb_words=8),
        ))
        shift_starts.append(start)
    session.flush()

    # --- Measurements + Alarms ---
    t0 = datetime.now() - timedelta(days=10)
    for tag, base, spread in tags:
        for i in range(MEASUREMENTS_PER_TAG):
            ts = t0 + timedelta(minutes=i * 5)
            # норма + редкие выбросы
            if random.random() < ALARM_RATE:
                value = base + random.choice([-1, 1]) * spread * random.uniform(4, 7)
            else:
                value = random.gauss(base, spread)
            quality = random.choices(["good", "uncertain", "bad"], weights=[95, 4, 1])[0]
            session.add(Measurement(tag_id=tag.id, ts=ts, value=round(value, 3), quality=quality))

            # alarm если вышли за пороги
            severity = None
            if value <= tag.lo_lo or value >= tag.hi_hi:
                severity = "critical"
            elif value <= tag.lo or value >= tag.hi:
                severity = "high"
            if severity:
                op = random.choice(operators)
                session.add(Alarm(
                    tag_id=tag.id, ts=ts, severity=severity,
                    message=f"{tag.parameter} вне границ: {round(value,2)} {tag.unit_of_measure}",
                    acknowledged_by=op.id,
                    acknowledged_at=ts + timedelta(minutes=random.randint(1, 30)),
                ))
        session.flush()

    # --- Maintenance ---
    for eq in equipments:
        for _ in range(random.randint(*MAINT_PER_EQUIP)):
            start = fake.date_time_between(start_date="-2y", end_date="now")
            dur = timedelta(hours=random.randint(2, 72))
            session.add(MaintenanceEvent(
                equipment_id=eq.id,
                started_at=start, ended_at=start + dur,
                type=random.choice(["planned", "unplanned", "inspection"]),
                description=fake.sentence(nb_words=10),
                performed_by=fake.name(),
            ))

    session.commit()


def main():
    engine = make_engine_and_schema()
    with Session(engine) as s:
        generate(s)
    print(f"Готово: {DB_URL}")


if __name__ == "__main__":
    main()
