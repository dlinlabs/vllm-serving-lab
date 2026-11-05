"""Finite anonymous-pipe framing, no hidden executor or socket."""

import asyncio
import json
import os
import selectors
import struct
import time

import common01 as c

HEADER = struct.Struct("!4sI32sddd")
MAGIC = b"CBC1"


def strict_json(raw):
    def pairs(rows):
        result = {}
        for key, value in rows:
            c.require(key not in result, "duplicate_JSON_key")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite_JSON")))


def frame(raw, *, sent, serial_started, serial_finished):
    c.require(type(raw) is bytes and 0 < len(raw) <= c.MAX_BYTES, "IPC_body_capacity")
    return HEADER.pack(MAGIC, len(raw), bytes.fromhex(c.sha(raw)),
                       sent, serial_started, serial_finished) + raw


def parse_header(raw):
    magic, size, digest, sent, started, finished = HEADER.unpack(raw)
    c.require(magic == MAGIC and 0 < size <= c.MAX_BYTES, "IPC_header")
    c.require(c.number(started) <= c.number(finished) <= c.number(sent), "IPC_serialization_chronology")
    return dict(size=size, sha256=digest.hex(), enqueue_send_started_at=sent,
                serialization_started_at=started, serialization_finished_at=finished)


def read_sync(fd, count, deadline, retained, *, first_read=None):
    output = bytearray()
    with selectors.DefaultSelector() as selector:
        selector.register(fd, selectors.EVENT_READ)
        while len(output) < count:
            remaining = deadline() - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("IPC_absolute_deadline")
            if not selector.select(remaining):
                raise TimeoutError("IPC_absolute_deadline")
            if first_read is not None and not first_read:
                first_read.append(time.monotonic())
            try:
                part = os.read(fd, min(65536, count - len(output)))
            except BlockingIOError:
                continue
            if not part:
                raise EOFError("IPC_partial_or_closed")
            retained.extend(part)
            output.extend(part)
    return bytes(output)


async def wait_fd(fd, *, writing, deadline):
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    def ready():
        if not future.done():
            future.set_result(None)
    register = loop.add_writer if writing else loop.add_reader
    remove = loop.remove_writer if writing else loop.remove_reader
    register(fd, ready)
    try:
        async with asyncio.timeout(max(0, deadline - time.monotonic())):
            await future
    finally:
        remove(fd)


async def write_async(fd, raw, deadline, receipt):
    offset = 0
    receipt.setdefault("written_bytes", 0)
    while offset < len(raw):
        if time.monotonic() >= deadline:
            raise TimeoutError("IPC_send_absolute_deadline")
        try:
            count = os.write(fd, raw[offset:offset + 65536])
        except BlockingIOError:
            await wait_fd(fd, writing=True, deadline=deadline)
            continue
        c.require(count > 0, "IPC_zero_write")
        offset += count
        receipt["written_bytes"] = offset


async def read_async(fd, count, deadline, retained):
    output = bytearray()
    while len(output) < count:
        if time.monotonic() >= deadline:
            raise TimeoutError("IPC_receive_absolute_deadline")
        try:
            part = os.read(fd, min(65536, count - len(output)))
        except BlockingIOError:
            await wait_fd(fd, writing=False, deadline=deadline)
            continue
        if not part:
            raise EOFError("IPC_partial_or_closed")
        retained.extend(part)
        output.extend(part)
    return bytes(output)


async def receive_async(fd, deadline, retained):
    header = parse_header(await read_async(fd, HEADER.size, deadline, retained))
    raw = await read_async(fd, header["size"], deadline, retained)
    c.require(c.sha(raw) == header["sha256"], "IPC_received_body_hash")
    return strict_json(raw), raw, header


def write_sync(fd, raw, deadline):
    sent = 0
    with selectors.DefaultSelector() as selector:
        selector.register(fd, selectors.EVENT_WRITE)
        while sent < len(raw):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("IPC_result_absolute_deadline")
            if not selector.select(remaining):
                raise TimeoutError("IPC_result_absolute_deadline")
            try:
                sent += os.write(fd, raw[sent:sent + 65536])
            except BlockingIOError:
                pass
