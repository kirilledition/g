//! Native bounded callback queue primitives.

use std::collections::VecDeque;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BoundedCallbackQueue<T> {
    capacity: usize,
    items: VecDeque<T>,
}

impl<T> BoundedCallbackQueue<T> {
    /// Create an empty queue with a fixed positive capacity.
    #[must_use]
    pub fn new(capacity: usize) -> Option<Self> {
        if capacity == 0 {
            return None;
        }
        Some(Self { capacity, items: VecDeque::with_capacity(capacity) })
    }

    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    #[must_use]
    pub fn occupied_count(&self) -> usize {
        self.items.len()
    }

    #[must_use]
    pub fn has_available_slot(&self) -> bool {
        self.items.len() < self.capacity
    }

    #[must_use]
    pub fn has_queued_item(&self) -> bool {
        !self.items.is_empty()
    }

    /// Attempt to append an item to the queue.
    ///
    /// # Errors
    ///
    /// Returns the original item when the queue is already at capacity.
    pub fn try_push(&mut self, item: T) -> Result<(), T> {
        if !self.has_available_slot() {
            return Err(item);
        }
        self.items.push_back(item);
        Ok(())
    }

    pub fn pop(&mut self) -> Option<T> {
        self.items.pop_front()
    }
}
