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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_callback_queue_tracks_capacity_and_fifo_order() {
        let mut queue = BoundedCallbackQueue::new(2).unwrap();

        assert_eq!(queue.capacity(), 2);
        assert_eq!(queue.occupied_count(), 0);
        assert!(queue.has_available_slot());
        assert!(!queue.has_queued_item());
        assert_eq!(queue.try_push("first"), Ok(()));
        assert_eq!(queue.try_push("second"), Ok(()));
        assert_eq!(queue.occupied_count(), 2);
        assert!(!queue.has_available_slot());
        assert_eq!(queue.try_push("third"), Err("third"));
        assert_eq!(queue.pop(), Some("first"));
        assert_eq!(queue.pop(), Some("second"));
        assert_eq!(queue.pop(), None);
    }

    #[test]
    fn bounded_callback_queue_rejects_zero_capacity() {
        assert_eq!(BoundedCallbackQueue::<usize>::new(0), None);
    }
}
