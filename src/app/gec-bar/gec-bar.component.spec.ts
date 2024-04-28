import { ComponentFixture, TestBed } from '@angular/core/testing';

import { GecBarComponent } from './gec-bar.component';

describe('GecBarComponent', () => {
  let component: GecBarComponent;
  let fixture: ComponentFixture<GecBarComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ GecBarComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(GecBarComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
