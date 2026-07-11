use std::io;
use std::sync::{Arc, Condvar, Mutex};

use tracing_appender::non_blocking::NonBlocking;
use tracing_subscriber::fmt::writer::{MakeWriter, OptionalWriter};

use super::TelemetryWriterFactory;
use super::line::TelemetryLineWriter;

static STDERR_WRITER: SharedWriterRegistry<NonBlocking> = SharedWriterRegistry::new("stderr");
static LOG_FILE_WRITER: SharedWriterRegistry<NonBlocking> = SharedWriterRegistry::new("log file");
static TELEMETRY_WRITER: SharedWriterRegistry<TelemetryWriterFactory> = SharedWriterRegistry::new("telemetry");

#[derive(Clone, Copy)]
pub(crate) enum SharedLogWriterKind {
    Stderr,
    File,
}

pub(crate) struct SharedLogWriterFactory {
    kind: SharedLogWriterKind,
}

pub(crate) struct SharedTelemetryWriterFactory;

pub(crate) struct SharedWriterLease<Writer, Factory> {
    writer: Writer,
    state: Arc<SharedWriterState<Factory>>,
}

struct SharedWriterRegistry<Factory> {
    name: &'static str,
    state: Mutex<Option<Arc<SharedWriterState<Factory>>>>,
}

struct SharedWriterState<Factory> {
    factory: Factory,
    active_writer_count: Mutex<usize>,
    no_active_writers: Condvar,
}

impl SharedLogWriterFactory {
    #[must_use]
    pub const fn new(kind: SharedLogWriterKind) -> Self {
        Self { kind }
    }
}

impl<'writer> MakeWriter<'writer> for SharedLogWriterFactory {
    type Writer = OptionalWriter<SharedWriterLease<NonBlocking, NonBlocking>>;

    fn make_writer(&'writer self) -> Self::Writer {
        let writer = match self.kind {
            SharedLogWriterKind::Stderr => STDERR_WRITER.acquire(Clone::clone),
            SharedLogWriterKind::File => LOG_FILE_WRITER.acquire(Clone::clone),
        };
        writer.ok().flatten().into()
    }
}

impl<'writer> MakeWriter<'writer> for SharedTelemetryWriterFactory {
    type Writer = OptionalWriter<SharedWriterLease<TelemetryLineWriter, TelemetryWriterFactory>>;

    fn make_writer(&'writer self) -> Self::Writer {
        TELEMETRY_WRITER.acquire(|writer| writer.make_writer()).ok().flatten().into()
    }
}

impl<Writer, Factory> io::Write for SharedWriterLease<Writer, Factory>
where
    Writer: io::Write,
{
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        self.writer.write(buffer)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.writer.flush()
    }
}

impl<Writer, Factory> Drop for SharedWriterLease<Writer, Factory> {
    fn drop(&mut self) {
        let Ok(mut active_writer_count) = self.state.active_writer_count.lock() else {
            return;
        };
        *active_writer_count = active_writer_count.saturating_sub(1);
        if *active_writer_count == 0 {
            self.state.no_active_writers.notify_all();
        }
    }
}

impl<Factory> SharedWriterRegistry<Factory> {
    const fn new(name: &'static str) -> Self {
        Self { name, state: Mutex::new(None) }
    }

    fn register(&self, factory: Factory) -> io::Result<()> {
        let mut state = self.lock_state()?;
        if state.is_some() {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!("A shared {} writer is already active in this process.", self.name),
            ));
        }
        *state = Some(Arc::new(SharedWriterState {
            factory,
            active_writer_count: Mutex::new(0),
            no_active_writers: Condvar::new(),
        }));
        Ok(())
    }

    fn unregister(&self) -> io::Result<()> {
        let state = self.lock_state()?.take();
        if let Some(state) = state {
            state.wait_for_active_writers()?;
        }
        Ok(())
    }

    fn acquire<Writer>(
        &self,
        make_writer: impl FnOnce(&Factory) -> Writer,
    ) -> io::Result<Option<SharedWriterLease<Writer, Factory>>> {
        self.lock_state()?.as_ref().map(|state| state.acquire(make_writer)).transpose()
    }

    fn lock_state(&self) -> io::Result<std::sync::MutexGuard<'_, Option<Arc<SharedWriterState<Factory>>>>> {
        self.state.lock().map_err(|_| io::Error::other(format!("Shared {} writer mutex was poisoned.", self.name)))
    }
}

impl<Factory> SharedWriterState<Factory> {
    fn acquire<Writer>(
        self: &Arc<Self>,
        make_writer: impl FnOnce(&Factory) -> Writer,
    ) -> io::Result<SharedWriterLease<Writer, Factory>> {
        let mut active_writer_count = self
            .active_writer_count
            .lock()
            .map_err(|_| io::Error::other("Shared active-writer mutex was poisoned."))?;
        *active_writer_count = active_writer_count.saturating_add(1);
        let writer = make_writer(&self.factory);
        drop(active_writer_count);
        Ok(SharedWriterLease { writer, state: Arc::clone(self) })
    }

    fn wait_for_active_writers(&self) -> io::Result<()> {
        let active_writer_count = self
            .active_writer_count
            .lock()
            .map_err(|_| io::Error::other("Shared active-writer mutex was poisoned."))?;
        let _active_writer_count = self
            .no_active_writers
            .wait_while(active_writer_count, |active_writer_count| *active_writer_count > 0)
            .map_err(|_| io::Error::other("Shared active-writer mutex was poisoned."))?;
        Ok(())
    }
}

/// Register one run-owned process log writer.
///
/// # Errors
///
/// Returns an I/O error when another run is active or the registry is poisoned.
pub fn register_shared_log_writer(kind: SharedLogWriterKind, writer: NonBlocking) -> io::Result<()> {
    match kind {
        SharedLogWriterKind::Stderr => STDERR_WRITER.register(writer),
        SharedLogWriterKind::File => LOG_FILE_WRITER.register(writer),
    }
}

/// Stop routing process logs to a run and wait for active formatters.
///
/// # Errors
///
/// Returns an I/O error when the registry or active-writer state is poisoned.
pub fn unregister_shared_log_writer(kind: SharedLogWriterKind) -> io::Result<()> {
    match kind {
        SharedLogWriterKind::Stderr => STDERR_WRITER.unregister(),
        SharedLogWriterKind::File => LOG_FILE_WRITER.unregister(),
    }
}

/// Register the run-owned telemetry writer used by the global logging layer.
///
/// # Errors
///
/// Returns an I/O error when another run is active or the registry is poisoned.
pub fn register_shared_telemetry_writer(writer: TelemetryWriterFactory) -> io::Result<()> {
    TELEMETRY_WRITER.register(writer)
}

/// Stop routing process logging to a run and wait for active formatters.
///
/// # Errors
///
/// Returns an I/O error when the registry or active-writer state is poisoned.
pub fn unregister_shared_telemetry_writer() -> io::Result<()> {
    TELEMETRY_WRITER.unregister()
}
