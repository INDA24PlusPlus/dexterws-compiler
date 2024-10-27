use std::{collections::HashMap, hash::Hash};

#[derive(Debug)]
pub struct ChainMap<K: Eq + Hash, V> {
    maps: Vec<HashMap<K, V>>,
}

impl<K, V> ChainMap<K, V>
where
    K: Eq + Hash,
{
    pub fn new() -> ChainMap<K, V> {
        ChainMap { maps: vec![] }
    }

    pub fn insert(&mut self, key: K, value: V) -> Option<()> {
        let map = self.maps.last_mut()?;
        map.insert(key, value);
        Some(())
    }

    pub fn get(&self, key: &K) -> Option<&V> {
        for map in self.maps.iter().rev() {
            if let Some(value) = map.get(key) {
                return Some(value);
            }
        }
        None
    }

    pub fn push(&mut self) {
        self.maps.push(HashMap::new());
    }

    pub fn pop(&mut self) -> Option<HashMap<K, V>> {
        self.maps.pop()
    }

    pub fn contains_key(&self, key: &K) -> bool {
        for map in self.maps.iter().rev() {
            if map.contains_key(key) {
                return true;
            }
        }
        false
    }
}
