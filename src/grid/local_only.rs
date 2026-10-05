//! The opt-out from [`GlobalTransform`] maintenance, and the per-grid list of the children that
//! keep it, so high-precision propagation visits only entities that still have a reader.

use crate::prelude::*;
use alloc::vec::Vec;
use bevy_ecs::{lifecycle::HookContext, prelude::*, world::DeferredWorld};
use bevy_reflect::prelude::*;
use bevy_transform::prelude::*;

/// Marks a high-precision entity whose [`GlobalTransform`] is not maintained: consumers read its
/// [`CellCoord`] and [`Transform`] against the grid instead.
///
/// Propagation never writes the marked entity's [`GlobalTransform`]; it keeps whatever it last
/// held, which is identity when the entity is spawned marked. Recentering still runs, so the
/// [`CellCoord`] and [`Transform`] stay valid. Low-precision descendants of a marked entity are not
/// propagated either, since the [`GlobalTransform`] they would inherit is stale. Removing the
/// marker recomputes the [`GlobalTransform`] at once and resumes maintenance.
///
/// The cost of a marked entity in propagation is zero per frame: it is absent from its grid's
/// [`PropagatedChildren`], so neither a moving floating origin nor a change to the entity visits it.
#[derive(Component, Debug, Default, Clone, Copy, Reflect)]
#[component(on_insert = GridLocalOnly::on_insert, on_discard = GridLocalOnly::on_discard)]
#[reflect(Component, Default)]
pub struct GridLocalOnly;

impl GridLocalOnly {
    fn on_insert(mut world: DeferredWorld, ctx: HookContext) {
        unlist(&mut world, ctx.entity);
    }

    /// The marker is still present while this hook runs, and a despawn also lands here, so the
    /// relist is deferred until the removal has completed.
    fn on_discard(mut world: DeferredWorld, ctx: HookContext) {
        let entity = ctx.entity;
        world.commands().queue(move |world: &mut World| {
            relist_unmarked(world, entity);
        });
    }
}

/// The children of a [`Grid`] whose [`GlobalTransform`] high-precision propagation maintains:
/// every child with a [`CellCoord`] and without [`GridLocalOnly`], in no particular order.
///
/// Maintained by component hooks and observers; required by [`Grid`].
#[derive(Component, Debug, Default)]
#[component(on_insert = PropagatedChildren::on_insert, on_discard = PropagatedChildren::on_discard)]
pub struct PropagatedChildren(Vec<Entity>);

impl PropagatedChildren {
    /// The listed entities.
    pub fn entities(&self) -> &[Entity] {
        &self.0
    }

    /// Lists the grid's existing children: the grid may gain this component after them.
    fn on_insert(mut world: DeferredWorld, ctx: HookContext) {
        let Some(children) = world.get::<Children>(ctx.entity) else {
            return;
        };
        let children: Vec<Entity> = children.iter().collect();
        for child in children {
            list_if_propagated(&mut world, child);
        }
    }

    fn on_discard(mut world: DeferredWorld, ctx: HookContext) {
        let Some(mut list) = world.get_mut::<PropagatedChildren>(ctx.entity) else {
            return;
        };
        let listed = core::mem::take(&mut list.0);
        for entity in listed {
            if let Some(mut slot) = world.get_mut::<PropagationSlot>(entity) {
                slot.0 = None;
            }
        }
    }
}

/// Where an entity sits in its grid's [`PropagatedChildren`], so listing is idempotent and
/// unlisting is a swap-remove instead of a search.
#[derive(Component, Debug, Default, Clone, Copy)]
pub(crate) struct PropagationSlot(Option<(Entity, u32)>);

/// Insert hook for [`CellCoord`].
pub(crate) fn list_on_insert(mut world: DeferredWorld, ctx: HookContext) {
    list_if_propagated(&mut world, ctx.entity);
}

/// Discard hook for [`CellCoord`]; an overwrite relists through the insert hook that follows.
pub(crate) fn unlist_on_discard(mut world: DeferredWorld, ctx: HookContext) {
    unlist(&mut world, ctx.entity);
}

/// Lists an entity whose parent changes; the [`CellCoord`] may already be present.
pub(crate) fn list_on_child_of_insert(trigger: On<Insert, ChildOf>, mut world: DeferredWorld) {
    list_if_propagated(&mut world, trigger.event_target());
}

/// Unlists an entity leaving its parent, before [`ChildOf`] is overwritten or removed.
pub(crate) fn unlist_on_child_of_discard(trigger: On<Discard, ChildOf>, mut world: DeferredWorld) {
    unlist(&mut world, trigger.event_target());
}

fn list_if_propagated(world: &mut DeferredWorld, entity: Entity) {
    let Ok(entity_ref) = world.get_entity(entity) else {
        return;
    };
    if entity_ref.contains::<GridLocalOnly>() || !entity_ref.contains::<CellCoord>() {
        return;
    }
    let (Some(parent), Some(slot)) = (
        entity_ref.get::<ChildOf>().map(ChildOf::parent),
        entity_ref.get::<PropagationSlot>().copied(),
    ) else {
        return;
    };
    if slot.0.is_some_and(|(grid, _)| grid == parent) {
        return;
    }
    unlist(world, entity);
    let Some(mut list) = world.get_mut::<PropagatedChildren>(parent) else {
        return;
    };
    // Entity indices are `u32`, so a list never outgrows one.
    let index = list.0.len() as u32;
    list.0.push(entity);
    if let Some(mut slot) = world.get_mut::<PropagationSlot>(entity) {
        slot.0 = Some((parent, index));
    }
}

fn unlist(world: &mut DeferredWorld, entity: Entity) {
    let Some(mut slot) = world.get_mut::<PropagationSlot>(entity) else {
        return;
    };
    let Some((grid, index)) = slot.0.take() else {
        return;
    };
    // The grid's list is gone when the grid lost its `Grid` or was despawned first.
    let Some(mut list) = world.get_mut::<PropagatedChildren>(grid) else {
        return;
    };
    let index = index as usize;
    debug_assert_eq!(
        list.0.get(index),
        Some(&entity),
        "propagation slot out of sync"
    );
    if list.0.get(index) != Some(&entity) {
        return;
    }
    list.0.swap_remove(index);
    let Some(&moved) = list.0.get(index) else {
        return;
    };
    if let Some(mut moved_slot) = world.get_mut::<PropagationSlot>(moved) {
        moved_slot.0 = Some((grid, index as u32));
    }
}

/// Relists an entity whose marker was removed and recomputes its stale [`GlobalTransform`] now,
/// since an unchanged entity under an unmoved origin is not recomputed by propagation.
fn relist_unmarked(world: &mut World, entity: Entity) {
    let Ok(entity_ref) = world.get_entity(entity) else {
        return;
    };
    if entity_ref.contains::<GridLocalOnly>() {
        return;
    }
    let (Some(parent), Some(cell), Some(transform)) = (
        entity_ref.get::<ChildOf>().map(ChildOf::parent),
        entity_ref.get::<CellCoord>().copied(),
        entity_ref.get::<Transform>().copied(),
    ) else {
        return;
    };
    list_if_propagated(&mut DeferredWorld::from(&mut *world), entity);
    let Some(global) = world
        .get::<Grid>(parent)
        .map(|grid| grid.global_transform(&cell, &transform))
    else {
        return;
    };
    if let Some(mut gt) = world.get_mut::<GlobalTransform>(entity) {
        *gt = global;
    }
}

#[cfg(test)]
mod tests {
    use super::PropagatedChildren;
    use crate::grid::propagation::LowPrecisionRoot;
    use crate::plugin::BigSpaceMinimalPlugins;
    use crate::prelude::*;
    use bevy::prelude::*;

    /// A root grid holding the floating origin; returns `(app, root, origin)`.
    fn app_with_origin() -> (App, Entity, Entity) {
        let mut app = App::new();
        app.add_plugins(BigSpaceMinimalPlugins);
        let root = app.world_mut().spawn(BigSpaceRootBundle::default()).id();
        let origin = app
            .world_mut()
            .spawn((CellCoord::default(), FloatingOrigin, ChildOf(root)))
            .id();
        (app, root, origin)
    }

    fn spawn_at(app: &mut App, root: Entity, x: f32, marked: bool) -> Entity {
        let mut entity = app.world_mut().spawn((
            CellCoord::default(),
            Transform::from_xyz(x, 0.0, 0.0),
            ChildOf(root),
        ));
        if marked {
            entity.insert(GridLocalOnly);
        }
        entity.id()
    }

    fn gt_x(app: &App, entity: Entity) -> f32 {
        app.world()
            .get::<GlobalTransform>(entity)
            .unwrap()
            .translation()
            .x
    }

    fn move_origin_one_cell(app: &mut App, origin: Entity) {
        app.world_mut().get_mut::<CellCoord>(origin).unwrap().x += 1;
    }

    fn listed(app: &App, grid: Entity) -> Vec<Entity> {
        let mut entities = app
            .world()
            .get::<PropagatedChildren>(grid)
            .unwrap()
            .entities()
            .to_vec();
        entities.sort();
        entities
    }

    fn sorted(mut entities: Vec<Entity>) -> Vec<Entity> {
        entities.sort();
        entities
    }

    #[test]
    fn marked_entity_keeps_its_gt_while_an_unmarked_one_follows_the_origin() {
        let (mut app, root, origin) = app_with_origin();
        let unmarked = spawn_at(&mut app, root, 50.0, false);
        let marked = spawn_at(&mut app, root, 50.0, true);
        app.update();
        assert_eq!(gt_x(&app, unmarked), 50.0);
        assert_eq!(
            gt_x(&app, marked),
            0.0,
            "a marked entity is never propagated"
        );

        move_origin_one_cell(&mut app, origin);
        app.update();
        assert_eq!(gt_x(&app, unmarked), 50.0 - 2000.0);
        assert_eq!(gt_x(&app, marked), 0.0);
    }

    #[test]
    fn removing_the_marker_recomputes_the_gt_without_origin_motion() {
        let (mut app, root, origin) = app_with_origin();
        let entity = spawn_at(&mut app, root, 50.0, true);
        move_origin_one_cell(&mut app, origin);
        app.update();
        app.update();
        assert_eq!(gt_x(&app, entity), 0.0);

        app.world_mut().entity_mut(entity).remove::<GridLocalOnly>();
        assert_eq!(
            gt_x(&app, entity),
            50.0 - 2000.0,
            "unmarking recomputes at once"
        );
        app.update();
        assert_eq!(gt_x(&app, entity), 50.0 - 2000.0);
        move_origin_one_cell(&mut app, origin);
        app.update();
        assert_eq!(gt_x(&app, entity), 50.0 - 4000.0, "maintenance resumes");
    }

    #[test]
    fn marking_stops_maintenance() {
        let (mut app, root, origin) = app_with_origin();
        let entity = spawn_at(&mut app, root, 50.0, false);
        app.update();
        app.world_mut().entity_mut(entity).insert(GridLocalOnly);
        move_origin_one_cell(&mut app, origin);
        app.update();
        assert_eq!(gt_x(&app, entity), 50.0);
    }

    #[test]
    fn marked_entity_still_recenters() {
        let (mut app, root, _) = app_with_origin();
        let entity = spawn_at(&mut app, root, 4100.0, true);
        app.update();
        let cell = *app.world().get::<CellCoord>(entity).unwrap();
        let local = app.world().get::<Transform>(entity).unwrap().translation.x;
        assert_eq!(
            (cell.x, local),
            (2, 100.0),
            "recentering still runs on a marked entity"
        );
        assert_eq!(gt_x(&app, entity), 0.0);
    }

    #[test]
    fn list_holds_exactly_the_unmarked_cell_children() {
        let (mut app, root, origin) = app_with_origin();
        let a = spawn_at(&mut app, root, 1.0, false);
        let b = spawn_at(&mut app, root, 2.0, false);
        let c = spawn_at(&mut app, root, 3.0, true);
        let plain = app
            .world_mut()
            .spawn((Transform::default(), ChildOf(root)))
            .id();
        assert_eq!(listed(&app, root), sorted(vec![origin, a, b]));

        app.world_mut().entity_mut(a).despawn();
        assert_eq!(
            listed(&app, root),
            sorted(vec![origin, b]),
            "despawn unlists"
        );

        app.world_mut().entity_mut(c).remove::<GridLocalOnly>();
        app.world_mut().entity_mut(b).insert(GridLocalOnly);
        assert_eq!(
            listed(&app, root),
            sorted(vec![origin, c]),
            "marker toggles move membership"
        );

        app.world_mut()
            .entity_mut(plain)
            .insert(CellCoord::default());
        app.world_mut()
            .entity_mut(c)
            .insert(CellCoord::new(1, 0, 0));
        assert_eq!(
            listed(&app, root),
            sorted(vec![origin, c, plain]),
            "a late or overwritten CellCoord lists once"
        );

        app.world_mut().entity_mut(c).despawn();
        app.world_mut().entity_mut(plain).remove::<CellCoord>();
        assert_eq!(
            listed(&app, root),
            vec![origin],
            "CellCoord removal unlists"
        );
    }

    #[test]
    fn reparenting_moves_an_entity_between_grid_lists() {
        let (mut app, root, origin) = app_with_origin();
        let sub = app
            .world_mut()
            .spawn((Grid::default(), CellCoord::default(), ChildOf(root)))
            .id();
        let entity = app
            .world_mut()
            .spawn((CellCoord::default(), Transform::from_xyz(7.0, 0.0, 0.0)))
            .id();
        assert!(listed(&app, sub).is_empty());
        app.world_mut().entity_mut(entity).insert(ChildOf(sub));
        assert_eq!(listed(&app, sub), vec![entity]);

        app.world_mut().entity_mut(entity).insert(ChildOf(root));
        assert!(listed(&app, sub).is_empty());
        assert_eq!(listed(&app, root), sorted(vec![origin, sub, entity]));
        app.update();
        assert_eq!(gt_x(&app, entity), 7.0);

        app.world_mut().entity_mut(entity).remove::<ChildOf>();
        assert_eq!(
            listed(&app, root),
            sorted(vec![origin, sub]),
            "orphaning unlists"
        );
    }

    #[test]
    fn grid_inserted_after_its_children_lists_them() {
        let mut app = App::new();
        app.add_plugins(BigSpaceMinimalPlugins);
        let parent = app.world_mut().spawn(BigSpace::default()).id();
        let origin = app
            .world_mut()
            .spawn((CellCoord::default(), FloatingOrigin, ChildOf(parent)))
            .id();
        spawn_at(&mut app, parent, 1.0, true);
        app.world_mut().entity_mut(parent).insert(Grid::default());
        assert_eq!(listed(&app, parent), vec![origin]);
    }

    #[test]
    fn children_of_a_marked_entity_are_not_low_precision_roots() {
        let (mut app, root, _) = app_with_origin();
        let marked = spawn_at(&mut app, root, 1.0, true);
        let marked_child = app
            .world_mut()
            .spawn((Transform::default(), ChildOf(marked)))
            .id();
        let unmarked = spawn_at(&mut app, root, 1.0, false);
        let unmarked_child = app
            .world_mut()
            .spawn((Transform::default(), ChildOf(unmarked)))
            .id();
        app.update();
        app.update();
        assert!(app.world().get::<LowPrecisionRoot>(marked_child).is_none());
        assert!(app
            .world()
            .get::<LowPrecisionRoot>(unmarked_child)
            .is_some());
    }

    /// Enough listed entities to take the fanned-out path, with as many marked ones beside them.
    #[test]
    fn channeled_path_skips_marked_entities() {
        let (mut app, root, origin) = app_with_origin();
        let count = 2 * Grid::INLINE_PROPAGATION_MAX;
        let unmarked: Vec<Entity> = (0..count)
            .map(|_| spawn_at(&mut app, root, 10.0, false))
            .collect();
        let marked: Vec<Entity> = (0..count)
            .map(|_| spawn_at(&mut app, root, 10.0, true))
            .collect();
        app.update();
        move_origin_one_cell(&mut app, origin);
        app.update();
        assert!(unmarked.iter().all(|&e| gt_x(&app, e) == 10.0 - 2000.0));
        assert!(marked.iter().all(|&e| gt_x(&app, e) == 0.0));
    }
}
