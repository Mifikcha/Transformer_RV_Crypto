from __future__ import annotations

import uuid

from sqlalchemy import Boolean, Column, Float, ForeignKey, Text
from sqlalchemy.dialects.postgresql import TIMESTAMP, UUID
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Bar5m(Base):
    __tablename__ = "bars_5m"

    ts = Column(TIMESTAMP(timezone=True), primary_key=True)
    open_perp = Column(Float)
    high_perp = Column(Float)
    low_perp = Column(Float)
    close_perp = Column(Float)
    volume_perp = Column(Float)
    turnover_perp = Column(Float)
    open_spot = Column(Float)
    high_spot = Column(Float)
    low_spot = Column(Float)
    close_spot = Column(Float)
    volume_spot = Column(Float)
    turnover_spot = Column(Float)
    funding_rate = Column(Float)
    open_interest = Column(Float)


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    ts = Column(TIMESTAMP(timezone=True), nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), server_default="now()")
    rv_3bar = Column(Float)
    rv_12bar = Column(Float)
    model_ver = Column(Text)
    degraded = Column(Boolean, nullable=False, server_default="false")


class RvActual(Base):
    __tablename__ = "rv_actual"

    ts = Column(TIMESTAMP(timezone=True), primary_key=True)
    rv_3bar = Column(Float)
    rv_12bar = Column(Float)


class NotificationLog(Base):
    __tablename__ = "notification_log"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    prediction_id = Column(UUID(as_uuid=True), ForeignKey("predictions.id"))
    sent_at = Column(TIMESTAMP(timezone=True), server_default="now()")
    alert_type = Column(Text)
