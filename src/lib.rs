
use std::hash::{Hash, BuildHasher, Hasher};
use rand::Rng;
use ahash::{AHasher, RandomState};
use roaring::RoaringTreemap; // we use the RoaringTreemap so we can have u64 support.
use std::ops::Add;
use std::fmt::Debug;
use std::borrow::BorrowMut;


const HASH_BITS: i64 = 64;

#[derive(Clone,Debug)]
struct Config {
    bucket_shift: i64,
    hash_number_shift: i64,
    shrink_factor: i64,
    rehash_threshold: f64,
    random_walk_coefficient: i64,
    stash_size: i64,
    bucket_length: i64,
    bucket_mask: i64,
    hash_mask: i64,
    hash_number: i64,
    log_size: i64
}

impl Default for Config {
    fn default() -> Self {
        Config {
            bucket_shift: 3,
            hash_number_shift: 2,
            shrink_factor: 0,
            rehash_threshold: 0.9,
            random_walk_coefficient: 2,
            stash_size: 2,
            bucket_length: 1 << 3, // the bit operators here are not needed, I added them as a reference to Utkan's implementation
            bucket_mask: (1 << 3) - 1,
            hash_mask: (1 << 2) - 1,
            hash_number: 1 << 2,
            log_size: 11 // default: 8 + bucket_shift
        }
    }
}

impl Config {
    fn new(bucket_shift: i64,
           hash_number_shift: i64,
           shrink_factor: i64,
           rehash_threshold: f64,
           random_walk_coefficient: i64,
           stash_size: i64,
           bucket_length: i64,
           bucket_mask: i64,
           hash_mask: i64,
           hash_number: i64,
           log_size: i64) -> Config {

        Config{
            bucket_shift,
            hash_number_shift,
            shrink_factor,
            rehash_threshold,
            random_walk_coefficient,
            stash_size,
            bucket_length,
            bucket_mask,
            hash_mask,
            hash_number,
            log_size
        }
    }
}
// Copy cannot be used as we use Vecs and Vecs live on the heap. Also, while we allow any Key that implements Hash to be inserted in the map
// the Bucket at the end stores keys as integers, this is because we expect the result of hash to be an int. Same for stash. Also, T needs to
// implement Clone which shouldn't be too restrictive.
#[derive(Clone,Debug)]
struct Bucket<K, T> {
    keys: Vec<Option<K>>,
    values: Vec<Option<T>>,
    index: RoaringTreemap // the bitmap will keep track of what cells are available or occupied. Instead of assuming 0 as a placeholder for that
}

impl<K: Clone + Debug, T: Clone + Debug> Bucket<K, T> {
    fn new(config: &Config) -> Bucket<K, T> {
        Bucket {
            keys: vec![None; 1 << config.bucket_shift],
            values: vec![None; 1 << config.bucket_shift],
            index: RoaringTreemap::new()
        }
    }

    fn stash(config: &Config) -> Bucket<K, T> {
        Bucket {
            keys: vec![None; config.stash_size as usize],
            values: vec![None; config.stash_size as usize],
            index: RoaringTreemap::new()
        }
    }

    fn add(&mut self, index: i64, key: K, value: T) {
        std::mem::replace(&mut self.values[index as usize], Some(value));
        std::mem::replace(&mut self.keys[index as usize], Some(key));
        index.add(index);
    }

    //replace is like add but we don't increment the index
    fn replace(&mut self, index: u64, key: K, value: T) {
        std::mem::replace(&mut self.values[index as usize], Some(value));
        std::mem::replace(&mut self.keys[index as usize], Some(key));
        index.add(index);

    }

    fn get(&mut self, index: u64) -> (Option<K>, Option<T>) {
        (self.keys[index as usize].clone(), self.values[index as usize].clone())
    }

    fn next_available_slot(&mut self) -> Option<u64> {
        self.index.max()
    }
}

#[derive(Clone, Debug)]
struct Cuckoo<K, T> {
    size: i64,
    buckets: Vec<Bucket<K, T>>,
    entries: i64,
    grow: i64,
    shrink: i64,
    rehash: i64,
    stash: Bucket<K, T>,
    e_item: bool,
    ekey: Option<K>,
    eval: Option<T>,
    seeds: Vec<(u64, u64, u64, u64)>,
    config: Config
}

impl<K: Hash + Clone + Debug + std::cmp::PartialEq, T: Clone + Debug> Cuckoo<K, T> {

    fn new(mut config: Config) -> Cuckoo<K, T> {
        let log_size = config.log_size - config.bucket_shift;
        if log_size <= 0 {
            config.log_size = 0;
        } else if log_size > 63 {
            panic!("Can't allocate more than 2^64 cells");
        } else if config.hash_number * config.hash_number_shift + config.bucket_shift + config.hash_number_shift > 63 {
            panic!("the greedy addition requires hash_number*hash_number_shift+bucket_shift+hash_number_shift < 63")
        } else if config.hash_number_shift > 8 {
            panic!("hash_number_shift is too large. reduce hash_number_shift.")
        }
        Cuckoo {
            size: log_size,
            buckets: vec![Bucket::new(&config); 1 << log_size],
            entries: 0,
            grow: 0,
            shrink: 0,
            rehash: 0,
            stash: Bucket::stash(&config),
            e_item: false,
            ekey: None,
            eval: None,
            seeds: {
                let mut seeds = vec![(0,0, 0, 0); config.hash_number as usize];
                let mut rng = rand::thread_rng();
                for x in seeds.iter_mut() {
                    *x = (rng.gen(), rng.gen(), rng.gen(), rng.gen());
                }
                seeds
            },
            config: config.clone()
        }
    }

    fn reseed(&mut self) {
        let mut rng = rand::thread_rng();
        for x in self.seeds.iter_mut() {
            *x = (rng.gen(), rng.gen(), rng.gen(), rng.gen());

        }
    }

    // implementation of the Fisher-Yates algorithm https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle
    fn shuffle(&mut self, hashes: &mut Vec<u64>, mut r: i64) {

        for j in (0..(hashes.len() - 1)).rev() {

            let i = (((r & self.config.hash_mask) as usize) % (j+1 as usize));

            let j_val = hashes.get(j).unwrap().clone();
            let i_val = hashes.get(i).unwrap().clone();

            std::mem::replace(&mut hashes[i as usize], j_val);
            std::mem::replace(&mut hashes[j as usize], i_val);

            r >>= self.config.hash_number_shift;
        }
    }

    fn len(self) -> i64 {
       self.entries.clone()
    }

    fn hashes(&mut self, key: &K) -> Vec<u64> {
        let mut vec = Vec::new();

        for i in 0..self.seeds.len() {
            let seed = self.seeds.get(i).unwrap();
            let hash = RandomState::with_seeds(seed.0, seed.1, seed.2, seed.3).build_hasher();
            vec.push(hash);
        }

        let mask = ((1 << self.size) - 1) as u64;
        let mut hashed_keys = Vec::new();

        for hasher in vec.iter_mut() {
            key.hash(hasher);
            hashed_keys.push(hasher.finish() & mask);
        }
        hashed_keys
    }

    fn delete(&mut self, key: K) {

        if self.try_delete(key) == false {
            return;
        }

        if (1 << (self.size+self.config.bucket_shift-self.config.shrink_factor)) > self.entries {
            for i in self.config.shrink_factor..0 {
                if self.grow(-i as i8) {
                    break
                }
            }
        }
    }

    fn try_delete(&mut self, key: K) -> bool {

        let hashes = self.hashes(&key);

        for hash in hashes {
            let buck = self.buckets.get_mut(hash as usize).unwrap();

            for (i,k) in buck.keys.iter_mut().enumerate() {
                match k {
                    Some(k) => {
                        if k.clone() == key {
                            self.entries -= 1;
                            buck.keys[i] = None;
                            buck.values[i] = None;
                            return true;
                        }
                    },
                    _ => ()
                }
            }
        }

        for (i,k) in self.stash.keys.iter_mut().enumerate() {
            match k {
                Some(k) => {
                    if k.clone() == key {
                        self.entries -= 1;
                        self.stash.keys[i] = None;
                        self.stash.values[i] = None;
                        return true;
                    }
                },
                _ => ()
            }
        }

        return false;
    }

    // this is the main function to add a new k,v to the map. it's built on a number of other functions
    // check below for their implementation. The main idea is that we try to insert and if we fail, we
    // try to grow the map until we succeed in adding the k,v
    fn put(&mut self, key: K, value: T) {

        loop {
           if self.try_insert(key.clone(), value.clone()) {
               return
           }
            let mut i = 1;
            if self.load_factor() < self.rehash as f64 {
                i = 0;
            }
            loop {
                if self.grow(i) {
                    break
                }
                i += 1;
            }
        }
    }

    fn try_insert(&mut self, key: K, value: T) -> bool {
        //first we check if we are updating an existing value
        let hashes = self.hashes(&key);


       let  (updated, free_slot, ibucket, index) = self.update(key.clone(),
                                                               value.clone(),
                                                               hashes.clone());
        if updated {
            return true;
        }

        //if not then let's check if we have a free slot
        if free_slot {
            self.add_at(key.clone(), value.clone(), ibucket, index);
            self.entries += 1;
            return true;
        }

        //nothing worked, let's get greedy
        if self.greedy_add(key.clone(), value.clone(), hashes.clone()){
          self.entries += 1;
            return true;
        }
        return false;
    }


    //check assignments below for default return values.
    fn update(&mut self, key: K, value: T, mut hashes: Vec<u64>) -> (bool,bool,i64,i64) {

        let mut updated = false;
        let mut free_slot = false;
        let mut bucket_index:i64 = -1;
        let mut index:i64 = -1;

        for (bi, bucket) in self.buckets.iter_mut().enumerate() {
            for (i, k) in bucket.keys.iter().enumerate() {
                match k {
                    Some(K) => {
                        let local_k = k.clone().unwrap();
                        if local_k == key {
                            bucket.values.insert(i, Some(value.clone()));
                            updated = true;
                            return (updated, free_slot, bucket_index, index);
                        }
                    },
                    None => {
                        if free_slot == false {
                            bucket_index = bi as i64;
                            index = i as i64;
                            free_slot = true;
                        }
                    }
                }
            }
        }

        for (i, k) in self.stash.keys.iter().enumerate() {
            match k {
                Some(K) => {
                    let local_k = k.clone().unwrap();
                    if local_k == key {
                        self.stash.values.insert(i, Some(value.clone()));
                        updated = true;
                        return (updated, free_slot, bucket_index, index);
                    }
                },
                _ => ()
            }

        }


        return (updated, free_slot,bucket_index,index)
    }

    fn add_at(&mut self, key: K, value: T, b_index: i64, index: i64) {
        let mut bucket = self.buckets.get_mut(b_index as usize).unwrap();
        bucket.add(index, key, value);
    }

    fn try_add(&mut self, key: K, value: T, hashes: Vec<u64>, ignore: bool, except: i64) -> bool {

        let hash = self.hashes(&key);
        for h in hash.iter() {
            if ignore && *h == except as u64 {
                continue
            }

            let buck = self.buckets.get_mut(*h as usize).unwrap(); // the hash

            match buck.next_available_slot() {
                None => {
                    buck.add(0 , key.clone(), value.clone());
                    return true
                },
                Some(ind) => {
                    buck.add((ind + 1) as i64, key.clone(), value.clone());
                    return true
                },
                _ => panic!("index out of bound kind of situation")
            }
        }
        false
    }

    fn greedy_add(&mut self, key: K, value: T, mut hashes: Vec<u64>) -> bool {

        let max = (1 + self.size) * self.config.random_walk_coefficient;
        let mut rng = rand::thread_rng();

        let mut old_hashes = hashes.clone();

        let mut local_key = key.clone();
        let mut local_value = value.clone();

        for step in 0..max {
           let mut r = rng.gen();
            self.shuffle(&mut hashes, r);
            r >>= self.config.hash_number * self.config.hash_number_shift;
            let i = r & self.config.bucket_mask;
            let d = (r >> self.config.bucket_shift) & self.config.hash_mask;

            let h = hashes.get(d as usize).unwrap().clone();
            let mut bucket = self.buckets.get_mut(h as usize).unwrap();


            let (e_key, e_val) = bucket.get(i as u64).clone();

            bucket.replace(i as u64, local_key.clone(), local_value.clone());

            match (e_key, e_val) {
                (None, None) => return true,
                (None, Some(v)) => return true,
                (Some(e), Some(v)) => {
                    if self.try_add(e.clone(),
                                    v.clone(),
                                    old_hashes.clone(),
                                    true, h as i64) { return true };

                    local_key = e.clone();
                    local_value = v.clone();
                    hashes = old_hashes.clone();
                }
                _ => {}
            }
        }

        let ind = self.stash.next_available_slot().unwrap();

        if ind < self.config.stash_size as u64 {
            self.stash.add(ind as i64, key, value);
            return true;
        }

        self.ekey = Some(key);
        self.eval = Some(value);
        self.e_item = true;

        return false;
    }

    // given a key, retrieve the value stored in the map
    fn get(&mut self, key: K) -> Option<T> {
        let hash = self.hashes(&key);

        for h in hash.iter() {
            let buck = self.buckets.get_mut(*h as usize).unwrap();
           for (i,k) in buck.keys.iter().enumerate() {
               match k {
                   Some(val) => {
                       return buck.values.get(i).unwrap().clone()
                   },
                   _ => ()
               }
           }
        }
        return None;
    }

    fn load_factor(&self) -> f64 {
        (self.entries / ( ( self.buckets.len() as i64) << self.config.hash_number_shift) ) as f64
    }

    fn grow(&mut self, delta: i8) -> bool {

        let mut self_cp = self.clone();
        self_cp.reseed();

       if delta == 0 {
           self_cp.rehash += 1;
       } else if delta > 0 {
           self_cp.grow += 1;
       }else {
           if self_cp.size <= 8 {
               return false;
           }
           self_cp.shrink += 1;
       }

        self_cp.size += delta as i64;

        if self_cp.size > HASH_BITS{
            panic!("Cuckoo: Cannot grow any further")
        }

        self_cp.buckets = vec![Bucket::new(&self_cp.config); 1 << self_cp.size];

        for buck  in self.buckets.iter() {
            for (i,k) in buck.keys.iter().enumerate() {
                match k {
                    None => continue,
                    Some(k) => {
                       let value = buck.values.get(i).unwrap().as_ref().unwrap(); //it should be safe to unwrap here, otherwise something very wrong I've done :/
                        let hashes = self_cp.hashes(k);

                        if self_cp.try_add(k.clone(), value.clone(), hashes.clone(),false, 0) {
                             continue
                         }

                        if self_cp.greedy_add(k.clone(), value.clone(), hashes.clone()) {
                            continue
                        }else {
                            return false;
                        }
                    }
                }
            }
        }

        if self_cp.e_item {
            let k = self_cp.ekey.as_ref().unwrap().clone();
            let v = self_cp.eval.as_ref().unwrap().clone();
            if self_cp.try_insert(k, v) {
                self_cp.e_item = false;
            } else {
                return false;
            }
        }

        // ok this is ugly :/
        self.buckets = self_cp.buckets.clone();
        self.size = self_cp.size;
        self.entries = self_cp.entries;
        self.grow = self_cp.grow;
        self.shrink = self_cp.shrink;
        self.rehash = self_cp.rehash;
        self.stash = self_cp.stash.clone();
        self.e_item = self_cp.e_item;
        self.ekey = self_cp.ekey;
        self.eval = self_cp.eval;
        self.seeds = self_cp.seeds.clone();
        self.config = self_cp.config.clone();

        return true
    }

}

#[cfg(test)]
mod tests {
    use super::Config;
    use super::Cuckoo;
    use ahash::{AHasher, RandomState};
    use std::hash::{Hasher, BuildHasher};
    use std::collections::HashSet;
    use rand::Rng;

    #[test]
    fn test_delete(){
        let mut conf_val = Config::default();
        let mut cuckoo: Cuckoo<i32, String> = Cuckoo::new(conf_val);

        let mut hash = cuckoo.hashes(&0);
        cuckoo.try_add(0 as i32, String::from("test"),hash.clone(), false, 0);
        assert_eq!(Some(String::from("test")), cuckoo.get(0 as i32));

        cuckoo.delete(0);
        assert_ne!(Some(String::from("test")), cuckoo.get(0 as i32));
    }


    #[test]
    fn test_try_delete(){
        let mut conf_val = Config::default();
        let mut cuckoo: Cuckoo<i32, String> = Cuckoo::new(conf_val);

        let mut hash = cuckoo.hashes(&0);
        cuckoo.try_add(0 as i32, String::from("test"),hash.clone(), false, 0);
        assert_eq!(Some(String::from("test")), cuckoo.get(0 as i32));

        cuckoo.try_delete(0);
        assert_ne!(Some(String::from("test")), cuckoo.get(0 as i32));
    }

    #[test]
    fn test_put(){
        let mut conf_val = Config::default();
        let mut cuckoo: Cuckoo<i32, String> = Cuckoo::new(conf_val);

        let mut hash = cuckoo.hashes(&0);
        cuckoo.try_add(0 as i32, String::from("test"),hash.clone(), false, 0);

        let mut cuckoo_upd = cuckoo.clone();
        cuckoo.update(0, String::from("test_updated"), hash.clone());

        assert_eq!(Some(String::from("test")), cuckoo_upd.get(0 as i32));
        assert_eq!(Some(String::from("test_updated")), cuckoo.get(0 as i32));
    }

    #[test]
    fn test_try_insert() {
        let mut conf_val = Config::default();
        let mut cuckoo: Cuckoo<i32, String> = Cuckoo::new(conf_val);

        let mut hash = cuckoo.hashes(&0);
        cuckoo.try_add(0 as i32, String::from("test"),hash.clone(), false, 0);

        let mut cuckoo_upd = cuckoo.clone();
        cuckoo.update(0, String::from("test_updated"), hash.clone());

        assert_eq!(Some(String::from("test")), cuckoo_upd.get(0 as i32));
        assert_eq!(Some(String::from("test_updated")), cuckoo.get(0 as i32));
    }

    #[test]
    fn test_update(){
        let mut conf_val = Config::default();
        let mut cuckoo: Cuckoo<i32, String> = Cuckoo::new(conf_val);

        let mut hash = cuckoo.hashes(&0);
        cuckoo.try_add(0 as i32, String::from("test"),hash.clone(), false, 0);

        let mut cuckoo_upd = cuckoo.clone();
        cuckoo.update(0, String::from("test_updated"), hash.clone());

        assert_eq!(Some(String::from("test")), cuckoo_upd.get(0 as i32));
        assert_eq!(Some(String::from("test_updated")), cuckoo.get(0 as i32));
    }


    #[test]
    fn test_grow(){
        let mut conf_val = Config::default();
        let mut cuckoo: Cuckoo<i32, String> = Cuckoo::new(conf_val);

        let mut hash = cuckoo.hashes(&0);
        cuckoo.try_add(0 as i32, String::from("test"),hash, false, 0);

        hash = cuckoo.hashes(&1);
        cuckoo.try_add(1 as i32, String::from("test1"),hash, false, 0);

        hash = cuckoo.hashes(&2);
        cuckoo.try_add(2 as i32, String::from("test2"),hash, false, 0);

        hash = cuckoo.hashes(&3);
        cuckoo.greedy_add(3 as i32, String::from("test3"),hash);

        hash = cuckoo.hashes(&4);
        cuckoo.greedy_add(4 as i32, String::from("test4"),hash);

        let t = cuckoo.clone();

        cuckoo.grow(2);

        println!("{:?}:{:?}",cuckoo.buckets.len(), t.buckets.len());
        assert_eq!(if cuckoo.buckets.len() > t.buckets.len(){ true}else {false}, true);
        assert_eq!(Some(String::from("test2")), cuckoo.get(2 as i32));
        assert_eq!(Some(String::from("test1")), cuckoo.get(1 as i32));
        assert_eq!(Some(String::from("test")), cuckoo.get(0 as i32));
        assert_eq!(Some(String::from("test3")), cuckoo.get(3 as i32));
        assert_eq!(Some(String::from("test4")), cuckoo.get(4 as i32));

    }

    #[test]
    fn test_greedy() {
        let mut conf_val = Config::default();
        let mut cuckoo: Cuckoo<i32, String> = Cuckoo::new(conf_val);

        let mut hash = cuckoo.hashes(&0);
        cuckoo.try_add(0 as i32, String::from("test"),hash, false, 0);

        hash = cuckoo.hashes(&1);
        cuckoo.greedy_add(1 as i32, String::from("test1"),hash);

        hash = cuckoo.hashes(&2);
        cuckoo.greedy_add(2 as i32, String::from("test2"),hash);

        assert_eq!(Some(String::from("test2")), cuckoo.get(2 as i32));
        assert_eq!(Some(String::from("test1")), cuckoo.get(1 as i32));
        assert_eq!(Some(String::from("test")), cuckoo.get(0 as i32));
    }

    #[test]
    fn test_shuffle() {
        let mut conf_val = Config::default();
        let mut cuckoo: Cuckoo<i32, String> = Cuckoo::new(conf_val);

        let mut hash = cuckoo.hashes(&0);
        let mut rng = rand::thread_rng();
        let hash1 = hash.clone();

        cuckoo.shuffle(&mut hash, rng.gen());

        assert_ne!(hash, hash1);

    }

    #[test]
    fn test_try_add() {
        let mut conf_val = Config::default();
        let mut cuckoo: Cuckoo<i32, String> = Cuckoo::new(conf_val);

        let hash = cuckoo.hashes(&0);
        cuckoo.try_add(0 as i32, String::from("test"),hash,false, 0);


        let hash1 = cuckoo.hashes(&2);
        cuckoo.try_add(2 as i32, String::from("test2"),hash1,false, 0);
        assert_eq!(Some(String::from("test2")), cuckoo.get(2 as i32));
        assert_eq!(Some(String::from("test")), cuckoo.get(0 as i32));
    }

    #[test]
    fn test_add_at() {
        let mut conf_val = Config::default();
        let mut cuckoo: Cuckoo<i32, String> = Cuckoo::new(conf_val);

        cuckoo.add_at(123 as i32, String::from("test"),1 , 5 );

        assert_eq!(cuckoo.buckets.get_mut(1 as usize).unwrap().get(5 as u64), (Some(123), Some(String::from("test"))))
    }

    #[test]
    fn test_hashes() {
        let mut conf_val = Config::default();
        let mut cuckoo: Cuckoo<i32, String> = Cuckoo::new(conf_val);
        let mut set = HashSet::new();
        let mut res = cuckoo.hashes(&123456);
        let mut res1 = cuckoo.hashes(&123456);

        assert_eq!(res, res1);
        assert_eq!(res.len(), cuckoo.config.hash_number as usize); // the hashes generated should have the same length as the configured length

        for x in &res {
            set.insert(x);
        }

        assert_eq!(res.len(), set.len());
    }

    #[test]
    fn test_bit_ops() {
        let mut conf_val = Config::default();
        let cuckoo: Cuckoo<i32, String> = Cuckoo::new(conf_val);
        assert_eq!(cuckoo.buckets.capacity(), 256)
    }

    #[test]
    fn test_reseed() {
        let mut conf_val = Config::default();
        let mut cuckoo: Cuckoo<i32, String> = Cuckoo::new(conf_val);
        let mut cuckoo1 = cuckoo.clone(); // we need to clone as we cannot implement copy here and borrow,
        cuckoo1.reseed();                                   //otherwise the compiler complains.

        assert_ne!(cuckoo.seeds, cuckoo1.seeds); //make sure we don't mess up the hashers and end up with the same hashed values.
    }

    #[test]
    fn test_hashers() {
        let mut conf_val = Config::default();
        let mut cuckoo: Cuckoo<i32, String> = Cuckoo::new(conf_val);

        let mut cuckoo1 = cuckoo.clone();
        cuckoo1.reseed();

        let val1 = cuckoo.hashes(&123456);
        let val2 = cuckoo.hashes(&1234567);
        let val3 = cuckoo.hashes(&123456);

        let val4 = cuckoo1.hashes(&123456);

        assert_ne!(val1, val2);
        assert_eq!(val1, val3);
        assert_ne!(val1, val4);

    }

    #[test]
    fn test_len(){
        let mut conf_val = Config::default();
        let mut cuckoo: Cuckoo<i32, String> = Cuckoo::new(conf_val);

        assert_eq!(cuckoo.len(), 0);
    }
}
