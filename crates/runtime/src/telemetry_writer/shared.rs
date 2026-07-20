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
    pub(crate) const fn new(kind: SharedLogWriterKind) -> Self {
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
pub(crate) fn register_shared_log_writer(kind: SharedLogWriterKind, writer: NonBlocking) -> io::Result<()> {
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
pub(crate) fn unregister_shared_log_writer(kind: SharedLogWriterKind) -> io::Result<()> {
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
pub(super) fn register_shared_telemetry_writer(writer: TelemetryWriterFactory) -> io::Result<()> {
    TELEMETRY_WRITER.register(writer)
}

/// Stop routing process logging to a run and wait for active formatters.
///
/// # Errors
///
/// Returns an I/O error when the registry or active-writer state is poisoned.
pub(super) fn unregister_shared_telemetry_writer() -> io::Result<()> {
    TELEMETRY_WRITER.unregister()
}

#[cfg(test)]
mod tests {
    use std::io::Write as _;
    use std::sync::mpsc;
    use std::time::{Duration, Instant};

    use super::*;

    #[test]
    fn registry_rejects_duplicates_routes_leases_and_waits_for_release() {
        let registry = Arc::new(SharedWriterRegistry::new("test"));
        assert!(registry.acquire(Clone::clone).expect("inactive registry should be readable").is_none());
        registry.register(String::from("writer")).expect("first factory should register");

        let duplicate_error = registry.register(String::from("duplicate")).expect_err("duplicate should fail");
        assert_eq!(duplicate_error.kind(), io::ErrorKind::AlreadyExists);
        assert!(duplicate_error.to_string().contains("shared test writer"));

        let lease = registry
            .acquire(Clone::clone)
            .expect("active registry should be readable")
            .expect("active registry should return a lease");
        assert_eq!(lease.writer, "writer");

        let (started_sender, started_receiver) = mpsc::channel();
        let (finished_sender, finished_receiver) = mpsc::channel();
        let unregister_registry = Arc::clone(&registry);
        let unregister_worker = std::thread::spawn(move || {
            started_sender.send(()).expect("unregister start should be observable");
            let result = unregister_registry.unregister();
            finished_sender.send(result).expect("unregister result should be observable");
        });
        started_receiver.recv().expect("unregister worker should start");
        let removal_deadline = Instant::now() + Duration::from_secs(1);
        loop {
            if registry.acquire(Clone::clone).expect("registry should remain readable").is_none() {
                break;
            }
            assert!(Instant::now() < removal_deadline, "unregister should remove the registry route");
            std::thread::yield_now();
        }
        assert!(finished_receiver.try_recv().is_err());
        drop(lease);
        finished_receiver
            .recv_timeout(Duration::from_secs(5))
            .expect("unregister should finish after lease release")
            .expect("unregister should succeed");
        unregister_worker.join().expect("unregister worker should complete");
        assert!(registry.acquire(Clone::clone).expect("unregistered registry should be readable").is_none());
    }

    #[test]
    fn shared_writer_lease_forwards_write_and_flush() {
        #[derive(Clone)]
        struct SharedBuffer {
            bytes: Arc<Mutex<Vec<u8>>>,
            flush_count: Arc<Mutex<usize>>,
        }

        impl io::Write for SharedBuffer {
            fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
                self.bytes.lock().expect("shared bytes should be available").extend_from_slice(buffer);
                Ok(buffer.len())
            }

            fn flush(&mut self) -> io::Result<()> {
                let mut flush_count = self.flush_count.lock().expect("flush counter should be available");
                *flush_count += 1;
                Ok(())
            }
        }

        let bytes = Arc::new(Mutex::new(Vec::new()));
        let flush_count = Arc::new(Mutex::new(0));
        let registry = SharedWriterRegistry::new("buffer");
        registry
            .register(SharedBuffer { bytes: Arc::clone(&bytes), flush_count: Arc::clone(&flush_count) })
            .expect("buffer factory should register");
        let mut lease = registry
            .acquire(Clone::clone)
            .expect("buffer registry should be readable")
            .expect("buffer registry should return a lease");
        assert_eq!(lease.write(b"record").expect("lease should forward write"), 6);
        lease.flush().expect("lease should forward flush");
        drop(lease);
        registry.unregister().expect("buffer registry should unregister");
        assert_eq!(&*bytes.lock().expect("shared bytes should be available"), b"record");
        assert_eq!(*flush_count.lock().expect("flush counter should be available"), 1);
    }

    #[test]
    fn poisoned_registry_and_active_writer_locks_return_io_errors() {
        let poisoned_registry = Arc::new(SharedWriterRegistry::new("poisoned"));
        let registry_for_worker = Arc::clone(&poisoned_registry);
        let _panic = std::thread::spawn(move || {
            let _guard = registry_for_worker.state.lock().expect("registry lock should begin healthy");
            panic!("poison registry lock");
        })
        .join();
        let registry_error = poisoned_registry.register(1_u8).expect_err("poisoned registry should fail");
        assert!(registry_error.to_string().contains("Shared poisoned writer mutex was poisoned"));

        let active_registry = SharedWriterRegistry::new("active");
        active_registry.register(1_u8).expect("active factory should register");
        let active_state = active_registry
            .state
            .lock()
            .expect("registry lock should be available")
            .as_ref()
            .cloned()
            .expect("registered state should exist");
        let state_for_worker = Arc::clone(&active_state);
        let _panic = std::thread::spawn(move || {
            let _guard = state_for_worker.active_writer_count.lock().expect("active-writer lock should begin healthy");
            panic!("poison active-writer lock");
        })
        .join();
        let acquire_error =
            active_registry.acquire(|factory| *factory).err().expect("poisoned active state should fail");
        assert_eq!(acquire_error.to_string(), "Shared active-writer mutex was poisoned.");
        let unregister_error = active_registry.unregister().expect_err("poisoned active state should not unregister");
        assert_eq!(unregister_error.to_string(), "Shared active-writer mutex was poisoned.");
    }
}
